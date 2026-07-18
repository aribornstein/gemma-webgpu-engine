import type { GemmaAudioGlobalWeights } from "../model/gemma-audio-weights";
import type { GemmaAudioFeatures } from "../runtime/gemma-audio-input";

const FEATURE_SIZE = 128;
const HIDDEN_SIZE = 1024;
const WORKGROUP_SIZE = 128;
const NORM_EPSILON = 1e-6;

export interface GemmaAudioSubsamplerPipelines {
  convolution: GPUComputePipeline;
  projection: GPUComputePipeline;
}

export interface GemmaAudioSubsamplerResources {
  output: GPUBuffer;
  outputMask: Uint32Array;
  outputRows: number;
  ownedBuffers: GPUBuffer[];
  convolution0BindGroup: GPUBindGroup;
  convolution1BindGroup: GPUBindGroup;
  projectionBindGroup: GPUBindGroup;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GemmaAudioSubsamplerPipelines>>();

export function getGemmaAudioSubsamplerPipelines(
  device: GPUDevice,
): Promise<GemmaAudioSubsamplerPipelines> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const pending = Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma audio subsampler convolution",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createGemmaAudioSubsamplerConvolutionShader() }),
        entryPoint: "main",
      },
    }),
    device.createComputePipelineAsync({
      label: "Gemma audio subsampler projection",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createGemmaAudioSubsamplerProjectionShader() }),
        entryPoint: "main",
      },
    }),
  ]).then(([convolution, projection]) => ({ convolution, projection })).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, pending);
  return pending;
}

export function createGemmaAudioSubsamplerResources(
  device: GPUDevice,
  pipelines: GemmaAudioSubsamplerPipelines,
  input: GemmaAudioFeatures,
  weights: GemmaAudioGlobalWeights["subsampler"],
): GemmaAudioSubsamplerResources {
  if (input.frameCount < 1 || input.features.length !== input.frameCount * FEATURE_SIZE ||
      input.mask.length !== input.frameCount) {
    throw new Error("Gemma audio subsampler input geometry is invalid");
  }
  validateWeights(weights);
  const stage0Rows = Math.ceil(input.frameCount / 2);
  const stage0Columns = FEATURE_SIZE / 2;
  const outputRows = Math.ceil(stage0Rows / 2);
  const outputColumns = stage0Columns / 2;
  const mask0 = Uint32Array.from(input.mask);
  const mask1 = subsampleMask(mask0);
  const outputMask = subsampleMask(mask1);
  const ownedBuffers: GPUBuffer[] = [];
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, size: number, usage = storage): GPUBuffer => {
    const buffer = device.createBuffer({ label, size, usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const upload = (
    label: string,
    values: Float32Array | Uint32Array,
    usage = storage,
  ): GPUBuffer => {
    const buffer = make(label, values.byteLength, usage);
    device.queue.writeBuffer(buffer, 0, values);
    return buffer;
  };
  const features = upload("Gemma audio features", input.features);
  const inputMask = upload("Gemma audio input mask", mask0);
  const stage0Mask = upload("Gemma audio stage 0 mask", mask1);
  const convolution0 = upload("Gemma audio convolution 0 weights", weights.convolution0);
  const norm0 = upload("Gemma audio convolution 0 norm", weights.norm0);
  const convolution1 = upload("Gemma audio convolution 1 weights", weights.convolution1);
  const norm1 = upload("Gemma audio convolution 1 norm", weights.norm1);
  const projection = upload("Gemma audio subsampler projection weights", weights.projection);
  const stage0 = make(
    "Gemma audio subsampler stage 0",
    stage0Rows * stage0Columns * 128 * 4,
  );
  const stage1 = make(
    "Gemma audio subsampler stage 1",
    outputRows * outputColumns * 32 * 4,
  );
  const output = make(
    "Gemma audio subsampler output",
    outputRows * HIDDEN_SIZE * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );
  const parameters0 = upload(
    "Gemma audio convolution 0 parameters",
    new Uint32Array([input.frameCount, FEATURE_SIZE, 1, 128]),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const parameters1 = upload(
    "Gemma audio convolution 1 parameters",
    new Uint32Array([stage0Rows, stage0Columns, 128, 32]),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const projectionParameters = upload(
    "Gemma audio subsampler projection parameters",
    new Uint32Array([outputRows, 0, 0, 0]),
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  return {
    output,
    outputMask,
    outputRows,
    ownedBuffers,
    convolution0BindGroup: device.createBindGroup({
      layout: pipelines.convolution.getBindGroupLayout(0),
      entries: [
        binding(0, features),
        binding(1, inputMask),
        binding(2, convolution0),
        binding(3, norm0),
        binding(4, stage0),
        binding(5, parameters0),
      ],
    }),
    convolution1BindGroup: device.createBindGroup({
      layout: pipelines.convolution.getBindGroupLayout(0),
      entries: [
        binding(0, stage0),
        binding(1, stage0Mask),
        binding(2, convolution1),
        binding(3, norm1),
        binding(4, stage1),
        binding(5, parameters1),
      ],
    }),
    projectionBindGroup: device.createBindGroup({
      layout: pipelines.projection.getBindGroupLayout(0),
      entries: [
        binding(0, stage1),
        binding(1, projection),
        binding(2, output),
        binding(3, projectionParameters),
      ],
    }),
  };
}

export function encodeGemmaAudioSubsampler(
  encoder: GPUCommandEncoder,
  pipelines: GemmaAudioSubsamplerPipelines,
  resources: GemmaAudioSubsamplerResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma audio subsampler" });
  pass.setPipeline(pipelines.convolution);
  pass.setBindGroup(0, resources.convolution0BindGroup);
  pass.dispatchWorkgroups(FEATURE_SIZE / 2, Math.ceil(resources.outputRows * 2));
  pass.setBindGroup(0, resources.convolution1BindGroup);
  pass.dispatchWorkgroups(FEATURE_SIZE / 4, resources.outputRows);
  pass.setPipeline(pipelines.projection);
  pass.setBindGroup(0, resources.projectionBindGroup);
  pass.dispatchWorkgroups(Math.ceil(resources.outputRows * HIDDEN_SIZE / WORKGROUP_SIZE));
  pass.end();
}

export function destroyGemmaAudioSubsamplerResources(
  resources: GemmaAudioSubsamplerResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

export function createGemmaAudioSubsamplerConvolutionShader(): string {
  return `struct Parameters {
  inputRows: u32,
  inputColumns: u32,
  inputChannels: u32,
  outputChannels: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> normWeights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> parameters: Parameters;
var<workgroup> values: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> sums: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let outputColumn = groupId.x;
  let outputRow = groupId.y;
  let outputColumns = (parameters.inputColumns + 1u) / 2u;
  let outputRows = (parameters.inputRows + 1u) / 2u;
  if (outputColumn >= outputColumns || outputRow >= outputRows) { return; }
  var value = 0.0;
  if (lane < parameters.outputChannels) {
    for (var inputChannel = 0u; inputChannel < parameters.inputChannels; inputChannel++) {
      for (var kernelRow = 0u; kernelRow < 3u; kernelRow++) {
        let sourceRow = i32(outputRow * 2u + kernelRow) - 1;
        if (sourceRow < 0 || sourceRow >= i32(parameters.inputRows) || mask[u32(sourceRow)] == 0u) {
          continue;
        }
        for (var kernelColumn = 0u; kernelColumn < 3u; kernelColumn++) {
          let sourceColumn = i32(outputColumn * 2u + kernelColumn) - 1;
          if (sourceColumn < 0 || sourceColumn >= i32(parameters.inputColumns)) { continue; }
          let inputIndex = (
            (u32(sourceRow) * parameters.inputColumns + u32(sourceColumn)) *
            parameters.inputChannels + inputChannel
          );
          let weightIndex = (
            ((lane * parameters.inputChannels + inputChannel) * 3u + kernelRow) *
            3u + kernelColumn
          );
          value = fma(input[inputIndex], weights[weightIndex], value);
        }
      }
    }
  }
  values[lane] = value;
  sums[lane] = value;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  let mean = sums[0] / f32(parameters.outputChannels);
  let centered = select(0.0, value - mean, lane < parameters.outputChannels);
  sums[lane] = centered * centered;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { sums[lane] += sums[lane + stride]; }
    workgroupBarrier();
  }
  if (lane < parameters.outputChannels) {
    let inverseStd = inverseSqrt(sums[0] / f32(parameters.outputChannels) + ${NORM_EPSILON});
    output[(outputRow * outputColumns + outputColumn) * parameters.outputChannels + lane] =
      max(0.0, centered * inverseStd * normWeights[lane]);
  }
}`;
}

export function createGemmaAudioSubsamplerProjectionShader(): string {
  return `struct Parameters { rows: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.rows * ${HIDDEN_SIZE}u) { return; }
  let row = index / ${HIDDEN_SIZE}u;
  let outputFeature = index % ${HIDDEN_SIZE}u;
  var accumulator = 0.0;
  for (var inputFeature = 0u; inputFeature < ${HIDDEN_SIZE}u; inputFeature++) {
    accumulator = fma(
      input[row * ${HIDDEN_SIZE}u + inputFeature],
      weights[outputFeature * ${HIDDEN_SIZE}u + inputFeature],
      accumulator,
    );
  }
  output[index] = accumulator;
}`;
}

function validateWeights(weights: GemmaAudioGlobalWeights["subsampler"]): void {
  if (weights.convolution0.length !== 128 * 1 * 3 * 3 ||
      weights.norm0.length !== 128 ||
      weights.convolution1.length !== 32 * 128 * 3 * 3 ||
      weights.norm1.length !== 32 ||
      weights.projection.length !== HIDDEN_SIZE * HIDDEN_SIZE) {
    throw new Error("Gemma audio subsampler weights do not match model geometry");
  }
}

function subsampleMask(mask: Uint32Array): Uint32Array {
  return Uint32Array.from({ length: Math.ceil(mask.length / 2) }, (_, index) => mask[index * 2]);
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}