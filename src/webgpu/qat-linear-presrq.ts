import { loadCapturedQatLinearFixture } from "../model/qat-linear-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 32;
const ROWS_PER_WORKGROUP = 2;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface PresrqQatLinearBenchmarkResult {
  operator: string;
  sourceOperator: "com.xenova.gemma4.QatMatMul";
  sourceVariant: "scalar_presrq";
  artifactSha256: string;
  referenceArtifactSha256: string;
  inFeatures: number;
  outFeatures: number;
  workgroupSize: number;
  rowsPerWorkgroup: number;
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
  maximumAbsoluteError: number;
  maximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  compileMs: number;
}

interface PresrqResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkCapturedQatLinearPresrq(
  iterations = 20,
): Promise<PresrqQatLinearBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const loaded = await loadCapturedQatLinearFixture();
  const { fixture, expectedOutput } = loaded;
  if (!fixture.inputSum || fixture.inputSum.length !== 1) {
    throw new Error("The captured presrq activation sum is unavailable");
  }

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

  const workgroupCount = Math.ceil(fixture.outFeatures / ROWS_PER_WORKGROUP);
  const resources = createResources(device, compiled.pipeline, fixture);
  try {
    await dispatch(device, compiled.pipeline, resources.bindGroup, workgroupCount);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, compiled.pipeline, resources.bindGroup, workgroupCount);
      dispatchSamples.push(performance.now() - started);
    }

    const encoder = device.createCommandEncoder({ label: "QatMatMul presrq readback" });
    encoder.copyBufferToBuffer(
      resources.outputBuffer,
      0,
      resources.readBuffer,
      0,
      expectedOutput.byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(resources.readBuffer.getMappedRange().slice(0));
    resources.readBuffer.unmap();

    const errors = measureErrors(actual, expectedOutput);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(
      device,
      compiled.pipeline,
      resources.bindGroup,
      workgroupCount,
    );

    return {
      operator: "model.language_model.layers.0.self_attn.q_proj",
      sourceOperator: "com.xenova.gemma4.QatMatMul",
      sourceVariant: "scalar_presrq",
      artifactSha256: loaded.containerSha256,
      referenceArtifactSha256: loaded.referenceSha256,
      inFeatures: fixture.inFeatures,
      outFeatures: fixture.outFeatures,
      workgroupSize: WORKGROUP_SIZE,
      rowsPerWorkgroup: ROWS_PER_WORKGROUP,
      workgroupCount,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      maximumAbsoluteError: errors.maximumAbsoluteError,
      maximumRelativeError: errors.maximumRelativeError,
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
  const module = device.createShaderModule({ code: createShader(subgroupReduction) });
  const pipeline = await device.createComputePipelineAsync({
    label: "QatMatMul scalar_presrq",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

function createShader(subgroupReduction: boolean): string {
  const reduction = subgroupReduction
    ? `enable subgroups;
fn reduce_sum(value: f32, lane: u32) -> f32 {
  return subgroupAdd(value);
}`
    : `var<workgroup> partial: array<f32, ${WORKGROUP_SIZE}>;
fn reduce_sum(value: f32, lane: u32) -> f32 {
  partial[lane] = value;
  workgroupBarrier();
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let result = partial[0];
  workgroupBarrier();
  return result;
}`;

  return `${reduction}
struct FloatData { values: array<f32> }
struct Float4Data { values: array<vec4<f32>> }
struct UintData { values: array<u32> }
struct Params {
  output_scale: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var<storage, read> activation: Float4Data;
@group(0) @binding(1) var<storage, read> bits: UintData;
@group(0) @binding(2) var<storage, read> scales: FloatData;
@group(0) @binding(3) var<storage, read> sum_a: FloatData;
@group(0) @binding(4) var<storage, read_write> output: FloatData;
@group(0) @binding(5) var<uniform> params: Params;

const IN_FEATURES: u32 = 1536u;
const OUT_FEATURES: u32 = 2048u;
const WORDS_PER_ROW: u32 = 192u;
const ZERO_POINT: f32 = 8.0;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lane = local_id.x;
  let row_base = workgroup_id.x * ${ROWS_PER_WORKGROUP}u;
  var sum0 = 0.0;
  var sum1 = 0.0;

  var word = lane;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    let a0 = activation.values[word * 2u];
    let a1 = activation.values[word * 2u + 1u];

    if (row_base < OUT_FEATURES) {
      let packed = bits.values[row_base * WORDS_PER_ROW + word];
      let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
      let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
      sum0 = sum0 + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), a0);
      sum0 = sum0 + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), a1);
    }
    if (row_base + 1u < OUT_FEATURES) {
      let packed = bits.values[(row_base + 1u) * WORDS_PER_ROW + word];
      let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
      let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
      sum1 = sum1 + dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), a0);
      sum1 = sum1 + dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), a1);
    }
    word = word + ${WORKGROUP_SIZE}u;
  }

  let reduced0 = reduce_sum(sum0, lane);
  let reduced1 = reduce_sum(sum1, lane);
  let zero_point_sum = ZERO_POINT * sum_a.values[0];
  if (lane == 0u && row_base < OUT_FEATURES) {
    output.values[row_base] = srq(
      scales.values[row_base] * (reduced0 - zero_point_sum),
      params.output_scale,
    );
  }
  if (lane == 0u && row_base + 1u < OUT_FEATURES) {
    output.values[row_base + 1u] = srq(
      scales.values[row_base + 1u] * (reduced1 - zero_point_sum),
      params.output_scale,
    );
  }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: Awaited<ReturnType<typeof loadCapturedQatLinearFixture>>["fixture"],
): PresrqResources {
  const inputBuffer = createBuffer(device, "presrq activation", fixture.input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "presrq packed weights", fixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scaleBuffer = createBuffer(device, "presrq row scales", fixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sumBuffer = createBuffer(device, "presrq activation sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "presrq Q output", fixture.outFeatures * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuffer = createBuffer(device, "presrq readback", fixture.outFeatures * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramsBuffer = createBuffer(device, "presrq parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const buffers = [inputBuffer, weightBuffer, scaleBuffer, sumBuffer, outputBuffer, readBuffer, paramsBuffer];

  device.queue.writeBuffer(inputBuffer, 0, fixture.input);
  device.queue.writeBuffer(weightBuffer, 0, fixture.packedWeights);
  device.queue.writeBuffer(scaleBuffer, 0, fixture.rowScales);
  device.queue.writeBuffer(sumBuffer, 0, fixture.inputSum!);
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([fixture.outputActivationScale ?? 0, 0, 0, 0]));

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: sumBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    }),
    outputBuffer,
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
  workgroupCount: number,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "QatMatMul presrq dispatch" });
  const pass = encoder.beginComputePass({ label: "QatMatMul scalar_presrq" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function measureGpuKernel(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroupCount: number,
): Promise<number[] | null> {
  if (!device.features.has("timestamp-query")) return null;
  const sampleCount = 10;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: "presrq timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "presrq timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "presrq timestamp sample" });
      const pass = encoder.beginComputePass({
        label: "QatMatMul scalar_presrq timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < DISPATCHES_PER_TIMESTAMP_SAMPLE; index += 1) {
        pass.dispatchWorkgroups(workgroupCount);
      }
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      if (sample >= 0) {
        samples.push(
          Number(timestamps[1] - timestamps[0]) /
            1e6 /
            DISPATCHES_PER_TIMESTAMP_SAMPLE,
        );
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
