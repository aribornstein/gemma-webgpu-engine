const WORKGROUP_SIZE = 128;

export interface GemmaAudioElementwisePipelines {
  silu: GPUComputePipeline;
  normalizedResidual: GPUComputePipeline;
}

export interface GemmaAudioSiluResources {
  bindGroup: GPUBindGroup;
  count: number;
  ownedBuffers: GPUBuffer[];
}

export interface GemmaAudioNormalizedResidualResources {
  bindGroup: GPUBindGroup;
  rows: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GemmaAudioElementwisePipelines>>();

export function getGemmaAudioElementwisePipelines(
  device: GPUDevice,
): Promise<GemmaAudioElementwisePipelines> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const pending = Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma audio SiLU",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createGemmaAudioSiluShader() }),
        entryPoint: "main",
      },
    }),
    device.createComputePipelineAsync({
      label: "Gemma audio normalized residual",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createGemmaAudioElementwiseShader() }),
        entryPoint: "normalizedResidual",
      },
    }),
  ]).then(([silu, normalizedResidual]) => ({ silu, normalizedResidual })).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, pending);
  return pending;
}

export function createGemmaAudioSiluResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  values: GPUBuffer,
  count: number,
): GemmaAudioSiluResources {
  if (!Number.isInteger(count) || count < 1 || values.size < count * 4) {
    throw new Error("Gemma audio SiLU buffer is too small");
  }
  const parameters = createParameters(device, new Uint32Array([count, 0, 0, 0]));
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [binding(0, values), binding(1, parameters)],
    }),
    count,
    ownedBuffers: [parameters],
  };
}

export function createGemmaAudioNormalizedResidualResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  input: GPUBuffer,
  weights: GPUBuffer,
  residual: GPUBuffer,
  rows: number,
  scale: number,
): GemmaAudioNormalizedResidualResources {
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 ||
      input.size < rows * 1024 * 4 || residual.size < rows * 1024 * 4 ||
      weights.size < 1024 * 4 || !Number.isFinite(scale)) {
    throw new Error("Gemma audio normalized residual buffers do not match model geometry");
  }
  const parameterBytes = new ArrayBuffer(16);
  const view = new DataView(parameterBytes);
  view.setUint32(0, rows, true);
  view.setFloat32(4, scale, true);
  const parameters = createParameters(device, parameterBytes);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        binding(0, input),
        binding(1, weights),
        binding(2, residual),
        binding(3, parameters),
      ],
    }),
    rows,
    ownedBuffers: [parameters],
  };
}

export function encodeGemmaAudioSilu(
  encoder: GPUCommandEncoder,
  pipelines: GemmaAudioElementwisePipelines,
  resources: GemmaAudioSiluResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma audio SiLU" });
  pass.setPipeline(pipelines.silu);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.count / 256));
  pass.end();
}

export function encodeGemmaAudioNormalizedResidual(
  encoder: GPUCommandEncoder,
  pipelines: GemmaAudioElementwisePipelines,
  resources: GemmaAudioNormalizedResidualResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma audio normalized residual" });
  pass.setPipeline(pipelines.normalizedResidual);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(resources.rows);
  pass.end();
}

export function createGemmaAudioElementwiseShader(): string {
  return `struct Parameters { count: u32, scale: f32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
var<workgroup> squareSums: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn normalizedResidual(
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = groupId.x;
  if (row >= parameters.count) { return; }
  var squareSum = 0.0;
  for (var column = lane; column < 1024u; column += ${WORKGROUP_SIZE}u) {
    let value = input[row * 1024u + column];
    squareSum = fma(value, value, squareSum);
  }
  squareSums[lane] = squareSum;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { squareSums[lane] += squareSums[lane + stride]; }
    workgroupBarrier();
  }
  let inverseRms = inverseSqrt(squareSums[0] / 1024.0 + 0.000001);
  for (var column = lane; column < 1024u; column += ${WORKGROUP_SIZE}u) {
    let index = row * 1024u + column;
    output[index] = fma(input[index] * inverseRms * weights[column], parameters.scale, output[index]);
  }
}`;
}

export function createGemmaAudioSiluShader(): string {
  return `struct Parameters { count: u32, scale: f32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> parameters: Parameters;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.count) { return; }
  let value = values[index];
  values[index] = value / (1.0 + exp(-value));
}`;
}

function createParameters(
  device: GPUDevice,
  values: ArrayBuffer | ArrayBufferView,
): GPUBuffer {
  const buffer = device.createBuffer({
    label: "Gemma audio elementwise parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, values as ArrayBuffer);
  return buffer;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}