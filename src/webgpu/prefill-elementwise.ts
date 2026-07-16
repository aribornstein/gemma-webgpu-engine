import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
  type GemmaPrefillParameterSlice,
} from "./prefill-parameter-arena";

export interface GemmaPrefillElementwisePipelines {
  add: GPUComputePipeline;
  multiply: GPUComputePipeline;
  geluMultiply: GPUComputePipeline;
  geluMultiplyStrided: GPUComputePipeline;
}

export interface GemmaPrefillElementwiseResources {
  bindGroup: GPUBindGroup;
  parameters: GemmaPrefillParameterSlice;
  count: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GemmaPrefillElementwisePipelines>>();

export function getGemmaPrefillElementwisePipelines(
  device: GPUDevice,
): Promise<GemmaPrefillElementwisePipelines> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = compileGemmaPrefillElementwisePipelines(device).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export async function compileGemmaPrefillElementwisePipelines(
  device: GPUDevice,
): Promise<GemmaPrefillElementwisePipelines> {
  const module = device.createShaderModule({ code: createGemmaPrefillElementwiseShader() });
  const [add, multiply, geluMultiply, geluMultiplyStrided] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma prefill residual add",
      layout: "auto",
      compute: { module, entryPoint: "add" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma prefill factor multiply",
      layout: "auto",
      compute: { module, entryPoint: "multiply" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma prefill LUT GELU multiply",
      layout: "auto",
      compute: { module, entryPoint: "geluMultiply" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma prefill strided LUT GELU multiply",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: createGemmaPrefillStridedGeluMultiplyShader(),
        }),
        entryPoint: "main",
      },
    }),
  ]);
  return { add, multiply, geluMultiply, geluMultiplyStrided };
}

export function createGemmaPrefillAddResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  destination: GPUBuffer,
  source: GPUBuffer,
  count: number,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillElementwiseResources {
  validateCount(count, destination, source);
  return createResources(
    device,
    pipeline,
    [destination, source],
    [count, 0, 0, 0],
    parameterArena,
  );
}

export function createGemmaPrefillMultiplyResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  values: GPUBuffer,
  factors: GPUBuffer,
  count: number,
  factorIndex: number,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillElementwiseResources {
  validateCount(count, values);
  if (!Number.isInteger(factorIndex) || factorIndex < 0 || factors.size < (factorIndex + 1) * 4) {
    throw new Error("Gemma prefill factor index exceeds its buffer");
  }
  return createResources(
    device,
    pipeline,
    [values, factors],
    [count, factorIndex, 0, 0],
    parameterArena,
  );
}

export function createGemmaPrefillGeluMultiplyResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  gate: GPUBuffer,
  multiplier: GPUBuffer,
  lookup: GPUBuffer,
  output: GPUBuffer,
  count: number,
  gridScale: number,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillElementwiseResources {
  validateCount(count, gate, multiplier, output);
  if (lookup.size < 256 * 4 || !Number.isFinite(gridScale) || gridScale <= 0) {
    throw new Error("Gemma prefill GELU lookup or grid scale is invalid");
  }
  return createResources(
    device,
    pipeline,
    [gate, multiplier, lookup, output],
    [count, 0, gridScale, 0],
    parameterArena,
  );
}

export function createGemmaPrefillStridedGeluMultiplyResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  gate: GPUBuffer,
  multiplier: GPUBuffer,
  lookup: GPUBuffer,
  output: GPUBuffer,
  rows: number,
  columns: number,
  sourceStride: number,
  sourceStart: number,
  gridScale: number,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillElementwiseResources {
  const values = [rows, columns, sourceStride, sourceStart];
  const count = rows * columns;
  const sourceEnd = (rows - 1) * sourceStride + sourceStart + columns;
  if (values.some((value) => !Number.isInteger(value) || value < 0) || rows < 1 ||
      columns < 1 || gate.size < count * 4 || multiplier.size < sourceEnd * 4 ||
      lookup.size < 256 * 4 || output.size < count * 4 ||
      !Number.isFinite(gridScale) || gridScale <= 0) {
    throw new Error("Gemma prefill strided GELU multiply geometry is invalid");
  }
  const parameterBuffer = createGemmaPrefillParameter(
    device,
    32,
    `${pipeline.label} parameters`,
    parameterArena,
  );
  const parameterBytes = new ArrayBuffer(32);
  const view = new DataView(parameterBytes);
  view.setUint32(0, count, true);
  view.setUint32(4, columns, true);
  view.setUint32(8, sourceStride, true);
  view.setUint32(12, sourceStart, true);
  view.setFloat32(16, gridScale, true);
  writeGemmaPrefillParameter(device, parameterBuffer.slice, parameterBytes);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gate } },
        { binding: 1, resource: { buffer: multiplier } },
        { binding: 2, resource: { buffer: lookup } },
        { binding: 3, resource: { buffer: output } },
        gemmaPrefillParameterBinding(4, parameterBuffer.slice),
      ],
    }),
    parameters: parameterBuffer.slice,
    count,
    ownedBuffers: parameterBuffer.ownedBuffers,
  };
}

export function encodeGemmaPrefillElementwise(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaPrefillElementwiseResources,
): void {
  const pass = encoder.beginComputePass({ label: pipeline.label });
  encodeGemmaPrefillElementwisePass(pass, pipeline, resources);
  pass.end();
}

export function encodeGemmaPrefillElementwisePass(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaPrefillElementwiseResources,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.count / 256));
}

export function destroyGemmaPrefillElementwiseResources(
  resources: GemmaPrefillElementwiseResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillElementwiseShader(): string {
  return `struct Parameters {
  count: u32,
  index: u32,
  scale: f32,
}

@group(0) @binding(0) var<storage, read_write> first: array<f32>;
@group(0) @binding(1) var<storage, read> second: array<f32>;
@group(0) @binding(2) var<storage, read> lookup: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

@compute @workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.count) { return; }
  first[index] = first[index] + second[index];
}

@compute @workgroup_size(256, 1, 1)
fn multiply(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.count) { return; }
  first[index] = first[index] * second[parameters.index];
}

@compute @workgroup_size(256, 1, 1)
fn geluMultiply(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.count) { return; }
  let lookupIndex = u32(clamp(round(first[index] / parameters.scale), -128.0, 127.0) + 128.0);
  output[index] = lookup[lookupIndex] * second[index];
}`;
}

export function createGemmaPrefillStridedGeluMultiplyShader(): string {
  return `struct Parameters {
  count: u32,
  columns: u32,
  sourceStride: u32,
  sourceStart: u32,
  scale: f32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> multiplier: array<f32>;
@group(0) @binding(2) var<storage, read> lookup: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.count) { return; }
  let row = index / parameters.columns;
  let column = index - row * parameters.columns;
  let sourceIndex = row * parameters.sourceStride + parameters.sourceStart + column;
  let lookupIndex = u32(
    clamp(round(gate[index] / parameters.scale), -128.0, 127.0) + 128.0,
  );
  output[index] = lookup[lookupIndex] * multiplier[sourceIndex];
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
  parameters: readonly number[],
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillElementwiseResources {
  const parameterBuffer = createGemmaPrefillParameter(
    device,
    16,
    `${pipeline.label} parameters`,
    parameterArena,
  );
  const parameterBytes = new ArrayBuffer(16);
  const view = new DataView(parameterBytes);
  view.setUint32(0, parameters[0], true);
  view.setUint32(4, parameters[1], true);
  view.setFloat32(8, parameters[2], true);
  writeGemmaPrefillParameter(device, parameterBuffer.slice, parameterBytes);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        gemmaPrefillParameterBinding(4, parameterBuffer.slice),
      ],
    }),
    parameters: parameterBuffer.slice,
    count: parameters[0],
    ownedBuffers: parameterBuffer.ownedBuffers,
  };
}

function validateCount(count: number, ...buffers: GPUBuffer[]): void {
  if (!Number.isInteger(count) || count < 1 || buffers.some((buffer) => buffer.size < count * 4)) {
    throw new Error("Gemma prefill elementwise count exceeds its buffers");
  }
}