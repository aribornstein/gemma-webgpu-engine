export interface GemmaPrefillElementwisePipelines {
  add: GPUComputePipeline;
  multiply: GPUComputePipeline;
  geluMultiply: GPUComputePipeline;
}

export interface GemmaPrefillElementwiseResources {
  bindGroup: GPUBindGroup;
  parameters: GPUBuffer;
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
  const [add, multiply, geluMultiply] = await Promise.all([
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
  ]);
  return { add, multiply, geluMultiply };
}

export function createGemmaPrefillAddResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  destination: GPUBuffer,
  source: GPUBuffer,
  count: number,
): GemmaPrefillElementwiseResources {
  validateCount(count, destination, source);
  return createResources(device, pipeline, [destination, source], [count, 0, 0, 0]);
}

export function createGemmaPrefillMultiplyResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  values: GPUBuffer,
  factors: GPUBuffer,
  count: number,
  factorIndex: number,
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
  );
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

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
  parameters: readonly number[],
): GemmaPrefillElementwiseResources {
  const parameterBuffer = device.createBuffer({
    label: `${pipeline.label} parameters`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const parameterBytes = new ArrayBuffer(16);
  const view = new DataView(parameterBytes);
  view.setUint32(0, parameters[0], true);
  view.setUint32(4, parameters[1], true);
  view.setFloat32(8, parameters[2], true);
  device.queue.writeBuffer(parameterBuffer, 0, parameterBytes);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: 4, resource: { buffer: parameterBuffer } },
      ],
    }),
    parameters: parameterBuffer,
    count: parameters[0],
    ownedBuffers: [parameterBuffer],
  };
}

function validateCount(count: number, ...buffers: GPUBuffer[]): void {
  if (!Number.isInteger(count) || count < 1 || buffers.some((buffer) => buffer.size < count * 4)) {
    throw new Error("Gemma prefill elementwise count exceeds its buffers");
  }
}