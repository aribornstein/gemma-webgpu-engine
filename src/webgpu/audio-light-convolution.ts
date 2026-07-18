const HIDDEN_SIZE = 1024;
const EXPANDED_SIZE = 2048;
const KERNEL_SIZE = 5;
const WORKGROUP_SIZE = 128;

export interface GemmaAudioLightConvolutionResources {
  bindGroup: GPUBindGroup;
  output: GPUBuffer;
  rows: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export function getGemmaAudioLightConvolutionPipeline(
  device: GPUDevice,
): Promise<GPUComputePipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const pending = device.createComputePipelineAsync({
    label: "Gemma audio fused light convolution",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: createGemmaAudioLightConvolutionShader() }),
      entryPoint: "main",
    },
  }).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, pending);
  return pending;
}

export function createGemmaAudioLightConvolutionResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  expandedInput: GPUBuffer,
  depthwiseWeights: GPUBuffer,
  normWeights: GPUBuffer,
  rows: number,
  output?: GPUBuffer,
): GemmaAudioLightConvolutionResources {
  const outputBytes = rows * HIDDEN_SIZE * 4;
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 ||
      expandedInput.size < rows * EXPANDED_SIZE * 4 ||
      depthwiseWeights.size < HIDDEN_SIZE * KERNEL_SIZE * 4 ||
      normWeights.size < HIDDEN_SIZE * 4 || (output && output.size < outputBytes)) {
    throw new Error("Gemma audio light convolution buffers do not match model geometry");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const outputBuffer = output ?? device.createBuffer({
    label: "Gemma audio light convolution output",
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  if (!output) ownedBuffers.push(outputBuffer);
  const parameters = device.createBuffer({
    label: "Gemma audio light convolution parameters",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(parameters, 0, new Uint32Array([rows, 0, 0, 0]));
  ownedBuffers.push(parameters);
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        binding(0, expandedInput),
        binding(1, depthwiseWeights),
        binding(2, normWeights),
        binding(3, outputBuffer),
        binding(4, parameters),
      ],
    }),
    output: outputBuffer,
    rows,
    ownedBuffers,
  };
}

export function encodeGemmaAudioLightConvolution(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  resources: GemmaAudioLightConvolutionResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma audio fused light convolution" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(resources.rows);
  pass.end();
}

export function destroyGemmaAudioLightConvolutionResources(
  resources: GemmaAudioLightConvolutionResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaAudioLightConvolutionShader(): string {
  return `struct Parameters { rows: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read> expandedInput: array<f32>;
@group(0) @binding(1) var<storage, read> depthwiseWeights: array<f32>;
@group(0) @binding(2) var<storage, read> normWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> parameters: Parameters;
var<workgroup> convolved: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> squareSums: array<f32, ${WORKGROUP_SIZE}>;

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = groupId.x;
  if (row >= parameters.rows) { return; }
  var squareSum = 0.0;
  for (var channel = lane; channel < ${HIDDEN_SIZE}u; channel += ${WORKGROUP_SIZE}u) {
    var value = 0.0;
    for (var kernel = 0u; kernel < ${KERNEL_SIZE}u; kernel++) {
      let sourceRow = i32(row) + i32(kernel) - ${KERNEL_SIZE - 1};
      if (sourceRow < 0) { continue; }
      let sourceBase = u32(sourceRow) * ${EXPANDED_SIZE}u;
      let gated = expandedInput[sourceBase + channel] *
        sigmoid(expandedInput[sourceBase + ${HIDDEN_SIZE}u + channel]);
      value = fma(gated, depthwiseWeights[channel * ${KERNEL_SIZE}u + kernel], value);
    }
    convolved[channel] = value;
    squareSum = fma(value, value, squareSum);
  }
  squareSums[lane] = squareSum;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { squareSums[lane] += squareSums[lane + stride]; }
    workgroupBarrier();
  }
  let inverseRms = inverseSqrt(squareSums[0] / ${HIDDEN_SIZE}.0 + 0.000001);
  for (var channel = lane; channel < ${HIDDEN_SIZE}u; channel += ${WORKGROUP_SIZE}u) {
    let normalized = convolved[channel] * inverseRms * normWeights[channel];
    output[row * ${HIDDEN_SIZE}u + channel] = normalized * sigmoid(normalized);
  }
}`;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}