import { cpuQatLinear, createQatLinearFixture, type QatLinearFixture } from "../reference/qat-linear";
import {
  loadCapturedQatLinearFixture,
  loadRealQatLinearFixture,
} from "../model/qat-linear-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 256;
const ABSOLUTE_TOLERANCE = 2e-5;
const RELATIVE_TOLERANCE = 2e-4;

const shader = `
struct Data { values: array<f32> }
struct PackedWeights { values: array<u32> }
struct Params {
  in_features: u32,
  out_features: u32,
  input_activation_scale: f32,
  output_activation_scale: f32,
  emulate_bfloat16: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: Data;
@group(0) @binding(1) var<storage, read> weights: PackedWeights;
@group(0) @binding(2) var<storage, read> row_scales: Data;
@group(0) @binding(3) var<storage, read_write> output: Data;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partials: array<f32, ${WORKGROUP_SIZE}>;

fn round_ties_to_even(value: f32) -> f32 {
  let lower = floor(value);
  let fraction = value - lower;
  if (fraction < 0.5) { return lower; }
  if (fraction > 0.5) { return lower + 1.0; }
  return select(lower + 1.0, lower, (i32(lower) & 1) == 0);
}

fn round_to_bfloat16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
  return bitcast<f32>(rounded & 0xffff0000u);
}

fn apply_srq(value: f32, scale: f32, emulate_bfloat16: bool) -> f32 {
  if (scale == 0.0) { return value; }
  let input_value = select(value, round_to_bfloat16(value), emulate_bfloat16);
  let effective_scale = select(scale, round_to_bfloat16(scale), emulate_bfloat16);
  var ratio = input_value / effective_scale;
  if (emulate_bfloat16) { ratio = round_to_bfloat16(ratio); }
  let quantized = clamp(round_ties_to_even(ratio), -128.0, 127.0);
  let output_value = quantized * effective_scale;
  return select(output_value, round_to_bfloat16(output_value), emulate_bfloat16);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = workgroup_id.x;
  if (row >= params.out_features) { return; }

  var partial = 0.0;
  let emulate_bfloat16 = params.emulate_bfloat16 != 0u;
  var column = lane;
  let row_offset = row * params.in_features;
  while (column < params.in_features) {
    let index = row_offset + column;
    let word = weights.values[index / 8u];
    let code = f32((word >> ((index & 7u) * 4u)) & 15u);
    let activation = apply_srq(
      input.values[column],
      params.input_activation_scale,
      emulate_bfloat16,
    );
    var weight = code - 8.0;
    if (emulate_bfloat16) {
      weight = round_to_bfloat16(weight * round_to_bfloat16(row_scales.values[row]));
    }
    partial += weight * activation;
    column += ${WORKGROUP_SIZE}u;
  }

  partials[lane] = partial;
  workgroupBarrier();
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (lane < stride) { partials[lane] += partials[lane + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }

  if (lane == 0u) {
    let projected = select(
      partials[0] * row_scales.values[row],
      partials[0],
      emulate_bfloat16,
    );
    output.values[row] = apply_srq(
      projected,
      params.output_activation_scale,
      emulate_bfloat16,
    );
  }
}`;

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  compileMs: number;
}

interface PipelineResult extends CompiledPipeline {
  cacheHit: boolean;
}

export interface QatLinearBenchmarkOptions {
  inFeatures?: number;
  outFeatures?: number;
  iterations?: number;
}

export interface QatLinearBenchmarkResult {
  operator: string;
  artifactSource: "synthetic" | "cached-export" | "buza-capture";
  artifactLoadMs: number;
  artifactSha256: string | null;
  referenceArtifactSha256: string | null;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
  bits: number;
  inFeatures: number;
  outFeatures: number;
  iterations: number;
  setupMs: number;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  dispatchAverageMs: number;
  gpuKernelDispatchesPerSample: number | null;
  gpuKernelMedianMs: number | null;
  gpuKernelP95Ms: number | null;
  readbackMs: number;
  cpuReferenceMs: number;
  maximumAbsoluteError: number;
  maximumRelativeError: number;
  absoluteTolerance: number;
  relativeTolerance: number;
  tolerancePassed: boolean;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

async function getPipeline(device: GPUDevice): Promise<PipelineResult> {
  const cached = pipelineCache.get(device);
  if (cached) return { ...(await cached), cacheHit: true };

  const compiled = (async (): Promise<CompiledPipeline> => {
    const started = performance.now();
    const module = device.createShaderModule({ code: shader });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return { pipeline, compileMs: performance.now() - started };
  })();
  pipelineCache.set(device, compiled);
  try {
    return { ...(await compiled), cacheHit: false };
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }
}

export async function benchmarkQatLinear(
  options: QatLinearBenchmarkOptions = {},
): Promise<QatLinearBenchmarkResult> {
  const inFeatures = options.inFeatures ?? 1536;
  const outFeatures = options.outFeatures ?? 2048;
  const iterations = options.iterations ?? 20;
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const fixture = createQatLinearFixture(inFeatures, outFeatures);
  return benchmarkFixture(fixture, iterations, {
    source: "synthetic",
    loadMs: 0,
    sha256: null,
    inputActivationScale: null,
    outputActivationScale: null,
  });
}

export async function benchmarkRealQatLinear(
  iterations = 20,
): Promise<QatLinearBenchmarkResult> {
  const loadStarted = performance.now();
  const loaded = await loadRealQatLinearFixture();
  const loadMs = performance.now() - loadStarted;
  return benchmarkFixture(loaded.fixture, iterations, {
    source: "cached-export",
    loadMs,
    sha256: loaded.containerSha256,
    referenceSha256: loaded.referenceSha256,
    expectedOutput: loaded.expectedOutput,
    inputActivationScale: loaded.inputActivationScale,
    outputActivationScale: loaded.outputActivationScale,
  });
}

export async function benchmarkCapturedQatLinear(
  iterations = 20,
): Promise<QatLinearBenchmarkResult> {
  const loadStarted = performance.now();
  const loaded = await loadCapturedQatLinearFixture();
  const loadMs = performance.now() - loadStarted;
  return benchmarkFixture(loaded.fixture, iterations, {
    source: "buza-capture",
    loadMs,
    sha256: loaded.containerSha256,
    referenceSha256: loaded.referenceSha256,
    expectedOutput: loaded.expectedOutput,
    inputActivationScale: loaded.inputActivationScale,
    outputActivationScale: loaded.outputActivationScale,
  });
}

interface ArtifactDetails {
  source: "synthetic" | "cached-export" | "buza-capture";
  loadMs: number;
  sha256: string | null;
  referenceSha256?: string | null;
  expectedOutput?: Float32Array;
  inputActivationScale: number | null;
  outputActivationScale: number | null;
}

async function benchmarkFixture(
  fixture: QatLinearFixture,
  iterations: number,
  artifact: ArtifactDetails,
): Promise<QatLinearBenchmarkResult> {
  const { inFeatures, outFeatures } = fixture;
  const cpuStarted = performance.now();
  const expected = artifact.expectedOutput ?? cpuQatLinear(fixture);
  const cpuReferenceMs = performance.now() - cpuStarted;
  const device = await getWebGpuDevice();
  const { pipeline, compileMs, cacheHit } = await getPipeline(device);
  const setupStarted = performance.now();
  const resources = createResources(device, pipeline, fixture);
  const setupMs = performance.now() - setupStarted;

  try {
    await dispatch(device, pipeline, resources.bindGroup, outFeatures);
    const dispatchSamples = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, pipeline, resources.bindGroup, outFeatures);
      dispatchSamples.push(performance.now() - started);
    }

    const readbackStarted = performance.now();
    const encoder = device.createCommandEncoder({ label: "QAT linear readback" });
    encoder.copyBufferToBuffer(
      resources.outputBuffer,
      0,
      resources.readBuffer,
      0,
      expected.byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(resources.readBuffer.getMappedRange().slice(0));
    resources.readBuffer.unmap();
    const readbackMs = performance.now() - readbackStarted;
    const errors = measureErrors(actual, expected);
    const sortedSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuKernelTiming = await measureGpuKernel(
      device,
      pipeline,
      resources.bindGroup,
      outFeatures,
    );

    return {
      operator: "model.language_model.layers.0.self_attn.q_proj",
      artifactSource: artifact.source,
      artifactLoadMs: round(artifact.loadMs),
      artifactSha256: artifact.sha256,
      referenceArtifactSha256: artifact.referenceSha256 ?? null,
      inputActivationScale: artifact.inputActivationScale,
      outputActivationScale: artifact.outputActivationScale,
      bits: 4,
      inFeatures,
      outFeatures,
      iterations,
      setupMs: round(setupMs),
      shaderCompilationMs: round(cacheHit ? 0 : compileMs),
      pipelineCacheHit: cacheHit,
      dispatchMedianMs: round(percentile(sortedSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedSamples, 0.95)),
      dispatchAverageMs: round(dispatchSamples.reduce((sum, value) => sum + value, 0) / iterations),
      gpuKernelDispatchesPerSample: gpuKernelTiming?.dispatchesPerSample ?? null,
      gpuKernelMedianMs: gpuKernelTiming
        ? round(percentile(gpuKernelTiming.samplesMs, 0.5))
        : null,
      gpuKernelP95Ms: gpuKernelTiming
        ? round(percentile(gpuKernelTiming.samplesMs, 0.95))
        : null,
      readbackMs: round(readbackMs),
      cpuReferenceMs: round(cpuReferenceMs),
      maximumAbsoluteError: errors.maximumAbsoluteError,
      maximumRelativeError: errors.maximumRelativeError,
      absoluteTolerance: ABSOLUTE_TOLERANCE,
      relativeTolerance: RELATIVE_TOLERANCE,
      tolerancePassed:
        errors.maximumAbsoluteError <= ABSOLUTE_TOLERANCE ||
        errors.maximumRelativeError <= RELATIVE_TOLERANCE,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

interface QatLinearResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: QatLinearFixture,
): QatLinearResources {
  const inputBuffer = createBuffer(device, "QAT input", fixture.input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "QAT packed weights", fixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scaleBuffer = createBuffer(device, "QAT row scales", fixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "QAT output", fixture.outFeatures * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuffer = createBuffer(device, "QAT readback", fixture.outFeatures * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramsBuffer = createBuffer(device, "QAT parameters", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const buffers = [inputBuffer, weightBuffer, scaleBuffer, outputBuffer, readBuffer, paramsBuffer];

  device.queue.writeBuffer(inputBuffer, 0, fixture.input);
  device.queue.writeBuffer(weightBuffer, 0, fixture.packedWeights);
  device.queue.writeBuffer(scaleBuffer, 0, fixture.rowScales);
  const params = new ArrayBuffer(32);
  const paramsView = new DataView(params);
  paramsView.setUint32(0, fixture.inFeatures, true);
  paramsView.setUint32(4, fixture.outFeatures, true);
  paramsView.setFloat32(8, fixture.inputActivationScale ?? 0, true);
  paramsView.setFloat32(12, fixture.outputActivationScale ?? 0, true);
  paramsView.setUint32(16, fixture.emulateBfloat16 ? 1 : 0, true);
  device.queue.writeBuffer(paramsBuffer, 0, params);

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    }),
    outputBuffer,
    readBuffer,
    buffers,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
  };
}

function createBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

async function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  outFeatures: number,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "QAT linear dispatch" });
  const pass = encoder.beginComputePass({ label: "Packed int4 matvec" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(outFeatures);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function measureGpuKernel(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  outFeatures: number,
): Promise<{ dispatchesPerSample: number; samplesMs: number[] } | null> {
  if (!device.features.has("timestamp-query")) return null;
  const dispatchesPerSample = 10;
  const sampleCount = 10;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: "QAT timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "QAT timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samplesMs: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "QAT timestamp sample" });
      const pass = encoder.beginComputePass({
        label: "Packed int4 matvec timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let dispatchIndex = 0; dispatchIndex < dispatchesPerSample; dispatchIndex += 1) {
        pass.dispatchWorkgroups(outFeatures);
      }
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      if (sample >= 0) {
        samplesMs.push(
          Number(timestamps[1] - timestamps[0]) / 1e6 / dispatchesPerSample,
        );
      }
    }
    return { dispatchesPerSample, samplesMs: samplesMs.toSorted((left, right) => left - right) };
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
  return sortedValues[Math.min(sortedValues.length - 1, Math.ceil(sortedValues.length * quantile) - 1)];
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}