import { loadDecodeRmsFixture, type DecodeRmsFixture } from "../model/decode-rms-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 256;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface DecodeRmsSrqBenchmarkResult {
  sourceOperator: "com.xenova.gemma4.DecodeRmsSrq";
  sourceVariant: "main";
  artifactSha256: string;
  sourceFixtureSha256: string;
  sourceCaptureSha256: string;
  hiddenSize: number;
  workgroupSize: number;
  workgroupCount: number;
  iterations: number;
  subgroupReduction: boolean;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  gpuKernelDispatchesPerSample: number | null;
  gpuKernelMedianMs: number | null;
  gpuKernelP95Ms: number | null;
  outputMaximumAbsoluteError: number;
  outputMaximumRelativeError: number;
  sumMaximumAbsoluteError: number;
  sumMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  compileMs: number;
}

interface DecodeRmsResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  sumBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkDecodeRmsSrq(
  iterations = 20,
): Promise<DecodeRmsSrqBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const fixture = await loadDecodeRmsFixture();
  const device = await getWebGpuDevice();
  const subgroupReduction = device.features.has("subgroups");
  const cached = pipelineCache.get(device);
  const compiledPromise = cached ?? compilePipeline(device, subgroupReduction);
  if (!cached) pipelineCache.set(device, compiledPromise);

  let compiled: CompiledPipeline;
  try {
    compiled = await compiledPromise;
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }

  const resources = createResources(device, compiled.pipeline, fixture);
  try {
    await dispatch(device, compiled.pipeline, resources.bindGroup);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, compiled.pipeline, resources.bindGroup);
      dispatchSamples.push(performance.now() - started);
    }

    const outputBytes = fixture.expectedOutput.byteLength;
    const encoder = device.createCommandEncoder({ label: "DecodeRmsSrq readback" });
    encoder.copyBufferToBuffer(resources.outputBuffer, 0, resources.readBuffer, 0, outputBytes);
    encoder.copyBufferToBuffer(resources.sumBuffer, 0, resources.readBuffer, outputBytes, 4);
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const mapped = resources.readBuffer.getMappedRange();
    const actualOutput = new Float32Array(mapped.slice(0, outputBytes));
    const actualSum = new Float32Array(mapped.slice(outputBytes, outputBytes + 4));
    resources.readBuffer.unmap();

    const outputErrors = measureErrors(actualOutput, fixture.expectedOutput);
    const sumErrors = measureErrors(actualSum, fixture.expectedSum);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(device, compiled.pipeline, resources.bindGroup);

    return {
      sourceOperator: "com.xenova.gemma4.DecodeRmsSrq",
      sourceVariant: "main",
      artifactSha256: fixture.artifactSha256,
      sourceFixtureSha256: fixture.sourceFixtureSha256,
      sourceCaptureSha256: fixture.sourceCaptureSha256,
      hiddenSize: fixture.hiddenSize,
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: 1,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      outputMaximumAbsoluteError: outputErrors.maximumAbsoluteError,
      outputMaximumRelativeError: outputErrors.maximumRelativeError,
      sumMaximumAbsoluteError: sumErrors.maximumAbsoluteError,
      sumMaximumRelativeError: sumErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipeline(
  device: GPUDevice,
  subgroupReduction: boolean,
): Promise<CompiledPipeline> {
  const started = performance.now();
  const module = device.createShaderModule({ code: createDecodeRmsSrqShader(subgroupReduction) });
  const pipeline = await device.createComputePipelineAsync({
    label: "DecodeRmsSrq main",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

export function createDecodeRmsSrqShader(subgroupReduction: boolean): string {
  const reduction = subgroupReduction
    ? `enable subgroups;
var<workgroup> subgroup_partial: array<f32, 8>;

fn reduce_sum(value: f32, thread: u32) -> f32 {
  let subgroup_sum = subgroupAdd(value);
  if ((thread & 31u) == 0u) {
    subgroup_partial[thread >> 5u] = subgroup_sum;
  }
  workgroupBarrier();
  var total = 0.0;
  for (var index = 0u; index < 8u; index = index + 1u) {
    total = total + subgroup_partial[index];
  }
  workgroupBarrier();
  return total;
}`
    : `var<workgroup> partial: array<f32, ${WORKGROUP_SIZE}>;

fn reduce_sum(value: f32, thread: u32) -> f32 {
  partial[thread] = value;
  workgroupBarrier();
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (thread < stride) {
      partial[thread] = partial[thread] + partial[thread + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let result = partial[0];
  workgroupBarrier();
  return result;
}`;

  return `${reduction}
struct Params {
  rows: u32,
  rowStride: u32,
  inScale: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> sum_a: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const DIM: u32 = 1536u;
const EPS: f32 = 0.000001;
const WG: u32 = ${WORKGROUP_SIZE}u;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row_stride = select(params.rowStride, params.rows, params.rowStride == 0u);
  let row = workgroup_id.x + workgroup_id.y * row_stride;
  if (row >= params.rows) { return; }
  let thread = local_id.x;
  let base = row * DIM;

  var square_sum = 0.0;
  var index = thread;
  loop {
    if (index >= DIM) { break; }
    let value = x[base + index];
    square_sum = square_sum + value * value;
    index = index + WG;
  }
  let scale = inverseSqrt(reduce_sum(square_sum, thread) / f32(DIM) + EPS);

  var quantized_sum = 0.0;
  index = thread;
  loop {
    if (index >= DIM) { break; }
    let normalized = x[base + index] * scale * w[index];
    let quantized = srq(normalized, params.inScale);
    y[base + index] = quantized;
    quantized_sum = quantized_sum + quantized;
    index = index + WG;
  }
  let total = reduce_sum(quantized_sum, thread);
  if (thread == 0u) {
    sum_a[row] = total;
  }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: DecodeRmsFixture,
): DecodeRmsResources {
  const inputBuffer = createBuffer(device, "DecodeRmsSrq input", fixture.hidden.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "DecodeRmsSrq weight", fixture.weight.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "DecodeRmsSrq output", fixture.expectedOutput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const sumBuffer = createBuffer(device, "DecodeRmsSrq sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuffer = createBuffer(device, "DecodeRmsSrq readback", fixture.expectedOutput.byteLength + 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramsBuffer = createBuffer(device, "DecodeRmsSrq parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const buffers = [inputBuffer, weightBuffer, outputBuffer, sumBuffer, readBuffer, paramsBuffer];

  const parameterBytes = new ArrayBuffer(16);
  const parameterView = new DataView(parameterBytes);
  parameterView.setUint32(0, 1, true);
  parameterView.setUint32(4, 1, true);
  parameterView.setFloat32(8, fixture.inputScale, true);
  device.queue.writeBuffer(inputBuffer, 0, fixture.hidden);
  device.queue.writeBuffer(weightBuffer, 0, fixture.weight);
  device.queue.writeBuffer(paramsBuffer, 0, parameterBytes);

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: sumBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    }),
    outputBuffer,
    sumBuffer,
    readBuffer,
    buffers,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
  };
}

function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

async function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "DecodeRmsSrq dispatch" });
  const pass = encoder.beginComputePass({ label: "DecodeRmsSrq main" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function measureGpuKernel(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
): Promise<number[] | null> {
  if (!device.features.has("timestamp-query")) return null;
  const sampleCount = 10;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: "DecodeRmsSrq timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "DecodeRmsSrq timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "DecodeRmsSrq timestamp sample" });
      const pass = encoder.beginComputePass({
        label: "DecodeRmsSrq timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < DISPATCHES_PER_TIMESTAMP_SAMPLE; index += 1) {
        pass.dispatchWorkgroups(1);
      }
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      if (sample >= 0) {
        samples.push(Number(timestamps[1] - timestamps[0]) / 1e6 / DISPATCHES_PER_TIMESTAMP_SAMPLE);
      }
    }
    return samples.toSorted((left, right) => left - right);
  } finally {
    querySet.destroy();
    resolveBuffer.destroy();
    readBuffer.destroy();
  }
}

function measureErrors(
  actual: Float32Array,
  expected: Float32Array,
): { maximumAbsoluteError: number; maximumRelativeError: number } {
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    const relative = absolute / Math.max(Math.abs(expected[index]), 1e-7);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absolute);
    maximumRelativeError = Math.max(maximumRelativeError, relative);
  }
  return { maximumAbsoluteError, maximumRelativeError };
}

function percentile(sortedValues: number[], quantile: number): number {
  return sortedValues[
    Math.min(sortedValues.length - 1, Math.ceil(sortedValues.length * quantile) - 1)
  ];
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}