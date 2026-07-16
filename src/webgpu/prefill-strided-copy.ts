import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
  type GemmaPrefillParameterSlice,
} from "./prefill-parameter-arena";

export interface GemmaPrefillStridedCopyParameters {
  rows: number;
  sourceStride: number;
  sourceStart: number;
  destinationStride: number;
  destinationStart: number;
  copyColumns: number;
}

export interface GemmaPrefillStridedCopyResources {
  bindGroup: GPUBindGroup;
  parameters: GemmaPrefillParameterSlice;
  source: GPUBuffer;
  destination: GPUBuffer;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaPrefillStridedCopyPipeline(
  device: GPUDevice,
): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = device.createComputePipelineAsync({
    label: "Gemma prefill strided copy",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaPrefillStridedCopyShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export function createGemmaPrefillStridedCopyResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  source: GPUBuffer,
  destination: GPUBuffer,
  parameters: GemmaPrefillStridedCopyParameters,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillStridedCopyResources {
  validateParameters(source, destination, parameters);
  const parameterBuffer = createGemmaPrefillParameter(
    device,
    32,
    "Gemma prefill strided copy parameters",
    parameterArena,
  );
  const resources = {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: destination } },
        gemmaPrefillParameterBinding(2, parameterBuffer.slice),
      ],
    }),
    parameters: parameterBuffer.slice,
    source,
    destination,
    ownedBuffers: parameterBuffer.ownedBuffers,
  };
  updateGemmaPrefillStridedCopy(device, resources, parameters);
  return resources;
}

export function updateGemmaPrefillStridedCopy(
  device: GPUDevice,
  resources: GemmaPrefillStridedCopyResources,
  parameters: GemmaPrefillStridedCopyParameters,
): void {
  validateParameters(resources.source, resources.destination, parameters);
  writeGemmaPrefillParameter(device, resources.parameters, new Uint32Array([
    parameters.rows,
    parameters.sourceStride,
    parameters.sourceStart,
    parameters.destinationStride,
    parameters.destinationStart,
    parameters.copyColumns,
    0,
    0,
  ]));
}

export function encodeGemmaPrefillStridedCopy(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaPrefillStridedCopyResources,
  rows: number,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma prefill strided copy" });
  encodeGemmaPrefillStridedCopyPass(pass, pipeline, resources, rows);
  pass.end();
}

export function encodeGemmaPrefillStridedCopyPass(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaPrefillStridedCopyResources,
  rows: number,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(rows);
}

export function destroyGemmaPrefillStridedCopyResources(
  resources: GemmaPrefillStridedCopyResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillStridedCopyShader(): string {
  return `struct Parameters {
  rows: u32,
  sourceStride: u32,
  sourceStart: u32,
  destinationStride: u32,
  destinationStart: u32,
  copyColumns: u32,
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = workgroupId.x;
  if (row >= parameters.rows) { return; }
  var column = lane;
  loop {
    if (column >= parameters.copyColumns) { break; }
    destination[row * parameters.destinationStride + parameters.destinationStart + column] =
      source[row * parameters.sourceStride + parameters.sourceStart + column];
    column = column + 64u;
  }
}`;
}

function validateParameters(
  source: GPUBuffer,
  destination: GPUBuffer,
  parameters: GemmaPrefillStridedCopyParameters,
): void {
  const values = Object.values(parameters);
  if (values.some((value) => !Number.isInteger(value) || value < 0) ||
      parameters.rows < 1 || parameters.copyColumns < 1) {
    throw new Error("Gemma prefill strided copy parameters must be non-negative integers");
  }
  const sourceEnd = (parameters.rows - 1) * parameters.sourceStride +
    parameters.sourceStart + parameters.copyColumns;
  const destinationEnd = (parameters.rows - 1) * parameters.destinationStride +
    parameters.destinationStart + parameters.copyColumns;
  if (sourceEnd * 4 > source.size || destinationEnd * 4 > destination.size) {
    throw new Error("Gemma prefill strided copy exceeds its source or destination buffer");
  }
}