import { loadDecodeOprojNormFixture } from "../model/decode-oproj-norm-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 256;
const OUT_FEATURES = 1536;
const WORKGROUP_COUNT = 192;

export interface DecodeOprojNormBenchmarkResult {
  sourceOperator: "com.xenova.gemma4.DecodeOprojNorm";
  sourceVariant: "fused-fixed-subgroup-32";
  artifactSha256: string;
  sourceMetadataSha256: string;
  sourceTensorsSha256: string;
  inFeatures: 2048;
  outFeatures: 1536;
  workgroupSize: 256;
  workgroupCount: 192;
  iterations: number;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  ffnInputBitMismatches: number;
  ffnInputSumMaximumAbsoluteError: number;
  ffnInputSumMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: 0;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
}

interface Resources {
  bindGroup: GPUBindGroup;
  hiddenBuffer: GPUBuffer;
  ffnInputBuffer: GPUBuffer;
  ffnInputSumBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkDecodeOprojNorm(
  iterations = 20,
): Promise<DecodeOprojNormBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }
  const [fixture, device] = await Promise.all([
    loadDecodeOprojNormFixture(),
    getWebGpuDevice(),
  ]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("DecodeOprojNorm requires WebGPU subgroups and shader-f16");
  }
  const compiledPromise = pipelineCache.get(device) ?? compilePipeline(device);
  if (!pipelineCache.has(device)) pipelineCache.set(device, compiledPromise);
  let compiled: CompiledPipeline;
  try {
    compiled = await compiledPromise;
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }

  const resources = createResources(device, compiled.pipeline, fixture);
  try {
    await resetAndDispatch(device, compiled.pipeline, resources, fixture.hiddenBefore);
    const samples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      device.queue.writeBuffer(resources.hiddenBuffer, 0, fixture.hiddenBefore);
      const started = performance.now();
      await dispatch(device, compiled.pipeline, resources.bindGroup);
      samples.push(performance.now() - started);
    }

    device.queue.writeBuffer(resources.hiddenBuffer, 0, fixture.hiddenBefore);
    await dispatch(device, compiled.pipeline, resources.bindGroup);
    const hiddenBytes = fixture.expectedHidden.byteLength;
    const ffnBytes = fixture.expectedFfnInputBits.byteLength;
    const sumBytes = fixture.expectedFfnInputSum.byteLength;
    const encoder = device.createCommandEncoder({ label: "DecodeOprojNorm readback" });
    encoder.copyBufferToBuffer(resources.hiddenBuffer, 0, resources.readBuffer, 0, hiddenBytes);
    encoder.copyBufferToBuffer(resources.ffnInputBuffer, 0, resources.readBuffer, hiddenBytes, ffnBytes);
    encoder.copyBufferToBuffer(
      resources.ffnInputSumBuffer,
      0,
      resources.readBuffer,
      hiddenBytes + ffnBytes,
      sumBytes,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const mapped = resources.readBuffer.getMappedRange();
    const actualHidden = new Float32Array(mapped.slice(0, hiddenBytes));
    const actualFfnBits = new Uint16Array(mapped.slice(hiddenBytes, hiddenBytes + ffnBytes));
    const actualSum = new Float32Array(mapped.slice(hiddenBytes + ffnBytes));
    resources.readBuffer.unmap();

    const hiddenErrors = measureErrors(actualHidden, fixture.expectedHidden);
    const sumErrors = measureErrors(actualSum, fixture.expectedFfnInputSum);
    let ffnInputBitMismatches = 0;
    for (let index = 0; index < actualFfnBits.length; index += 1) {
      if (actualFfnBits[index] !== fixture.expectedFfnInputBits[index]) {
        ffnInputBitMismatches += 1;
      }
    }
    const sortedSamples = samples.toSorted((left, right) => left - right);
    return {
      sourceOperator: "com.xenova.gemma4.DecodeOprojNorm",
      sourceVariant: "fused-fixed-subgroup-32",
      artifactSha256: fixture.artifactSha256,
      sourceMetadataSha256: fixture.sourceMetadataSha256,
      sourceTensorsSha256: fixture.sourceTensorsSha256,
      inFeatures: fixture.inFeatures,
      outFeatures: fixture.outFeatures,
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: WORKGROUP_COUNT,
      iterations,
      dispatchMedianMs: round(percentile(sortedSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedSamples, 0.95)),
      hiddenMaximumAbsoluteError: hiddenErrors.maximumAbsoluteError,
      hiddenMaximumRelativeError: hiddenErrors.maximumRelativeError,
      ffnInputBitMismatches,
      ffnInputSumMaximumAbsoluteError: sumErrors.maximumAbsoluteError,
      ffnInputSumMaximumRelativeError: sumErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<CompiledPipeline> {
  const module = device.createShaderModule({ code: createDecodeOprojNormShader() });
  const pipeline = await device.createComputePipelineAsync({
    label: "DecodeOprojNorm fixed subgroup 32",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline };
}

export function createDecodeOprojNormShader(
  inFeatures: 2048 | 4096 = 2048,
): string {
  const wordsPerRow = inFeatures / 8;
  return `enable f16;
enable subgroups;

struct Params {
  outScale: f32,
  inScale2: f32,
}

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits_buf: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read> w12: array<f32>;
@group(0) @binding(5) var<storage, read_write> pp: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> y2: array<f16>;
@group(0) @binding(7) var<storage, read_write> sum2: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const IN_FEATURES: u32 = ${inFeatures}u;
const OUT_F: u32 = 1536u;
const WORDS_PER_ROW: u32 = ${wordsPerRow}u;
const WG: u32 = 256u;
const SG_ROWS: u32 = 1u;
const ROWS_PER_WG: u32 = 8u;
const TOTAL_WGS: u32 = 192u;
const EPS: f32 = 0.000001;
const ELEMS: u32 = 6u;
const ZP: f32 = 8.0;

var<workgroup> lastFlag: u32;
var<workgroup> sgp: array<f32, 8>;

fn sg_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}

fn reduce_sum(value: f32, tid: u32) -> f32 {
  let subgroup_sum = sg_sum(value);
  if ((tid & 31u) == 0u) { sgp[tid >> 5u] = subgroup_sum; }
  workgroupBarrier();
  var total = 0.0;
  for (var index = 0u; index < 8u; index = index + 1u) {
    total = total + sgp[index];
  }
  workgroupBarrier();
  return total;
}

fn srq(value: f32, value_scale: f32) -> f32 {
  if (value_scale == 0.0) { return value; }
  return clamp(round(value / value_scale), -128.0, 127.0) * value_scale;
}

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let tid = local_id.x;
  let subgroup_id = tid / 32u;
  let lane = tid & 31u;
  let row_base = workgroup_id.x * ROWS_PER_WG + subgroup_id * SG_ROWS;
  var sum_qa: array<f32, 1>;
  sum_qa[0] = 0.0;
  var sum_a = 0.0;
  var word = lane;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    let activation0 = a[word * 2u];
    let activation1 = a[word * 2u + 1u];
    sum_a = sum_a + activation0.x + activation0.y + activation0.z + activation0.w;
    sum_a = sum_a + activation1.x + activation1.y + activation1.z + activation1.w;
    let output_row = row_base;
    if (output_row < OUT_F) {
      let packed = bits_buf[output_row * WORDS_PER_ROW + word];
      let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
      let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
      sum_qa[0] = sum_qa[0] +
        dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), activation0) +
        dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), activation1);
    }
    word = word + 32u;
  }
  let reduced_a = sg_sum(sum_a);
  let reduced_qa = sg_sum(sum_qa[0]);
  let output_row = row_base;
  if (lane == 0u && output_row < OUT_F) {
    atomicStore(&pp[output_row], bitcast<u32>(srq(
      scale[output_row] * (reduced_qa - ZP * reduced_a),
      params.outScale,
    )));
  }
  storageBarrier();

  if (tid == 0u) {
    let ticket = atomicAdd(&pp[OUT_F], 1u);
    lastFlag = select(0u, 1u, ticket == TOTAL_WGS - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) { return; }
  if (tid == 0u) { atomicStore(&pp[OUT_F], 0u); }

  var first_accumulator = 0.0;
  var index = tid;
  loop {
    if (index >= OUT_F) { break; }
    let value = bitcast<f32>(atomicLoad(&pp[index]));
    first_accumulator = first_accumulator + value * value;
    index = index + WG;
  }
  let first_rms = inverseSqrt(reduce_sum(first_accumulator, tid) / f32(OUT_F) + EPS);

  var local_hidden: array<f32, 6>;
  var second_accumulator = 0.0;
  var element = 0u;
  index = tid;
  loop {
    if (index >= OUT_F) { break; }
    let normalized = bitcast<f32>(atomicLoad(&pp[index])) * first_rms * w12[index];
    let hidden_value = hidden[index] + normalized;
    hidden[index] = hidden_value;
    local_hidden[element] = hidden_value;
    second_accumulator = second_accumulator + hidden_value * hidden_value;
    index = index + WG;
    element = element + 1u;
  }
  let second_rms = inverseSqrt(reduce_sum(second_accumulator, tid) / f32(OUT_F) + EPS);

  var quantized_sum = 0.0;
  index = tid;
  element = 0u;
  loop {
    if (index >= OUT_F) { break; }
    let normalized = local_hidden[element] * second_rms * w12[OUT_F + index];
    let quantized = f16(srq(f32(f16(normalized)), params.inScale2));
    y2[index] = quantized;
    quantized_sum = quantized_sum + f32(quantized);
    index = index + WG;
    element = element + 1u;
  }
  let total = reduce_sum(quantized_sum, tid);
  if (tid == 0u) { sum2[0] = total; }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: Awaited<ReturnType<typeof loadDecodeOprojNormFixture>>,
): Resources {
  const attentionBuffer = createBuffer(device, "DecodeOprojNorm attention", fixture.attention.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightsBuffer = createBuffer(device, "DecodeOprojNorm packed weights", fixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scalesBuffer = createBuffer(device, "DecodeOprojNorm row scales", fixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const hiddenBuffer = createBuffer(device, "DecodeOprojNorm hidden", fixture.hiddenBefore.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const normWeightsBuffer = createBuffer(device, "DecodeOprojNorm norm weights", fixture.normWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const partialBuffer = createBuffer(device, "DecodeOprojNorm partials", (OUT_FEATURES + 1) * 4, GPUBufferUsage.STORAGE);
  const ffnInputBuffer = createBuffer(device, "DecodeOprojNorm FFN input", fixture.expectedFfnInputBits.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const ffnInputSumBuffer = createBuffer(device, "DecodeOprojNorm FFN sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const paramsBuffer = createBuffer(device, "DecodeOprojNorm parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readBuffer = createBuffer(device, "DecodeOprojNorm readback", fixture.expectedHidden.byteLength + fixture.expectedFfnInputBits.byteLength + 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [attentionBuffer, weightsBuffer, scalesBuffer, hiddenBuffer, normWeightsBuffer, partialBuffer, ffnInputBuffer, ffnInputSumBuffer, paramsBuffer, readBuffer];

  device.queue.writeBuffer(attentionBuffer, 0, fixture.attention);
  device.queue.writeBuffer(weightsBuffer, 0, fixture.packedWeights);
  device.queue.writeBuffer(scalesBuffer, 0, fixture.rowScales);
  device.queue.writeBuffer(hiddenBuffer, 0, fixture.hiddenBefore);
  device.queue.writeBuffer(normWeightsBuffer, 0, fixture.normWeights);
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([fixture.outputScale, fixture.inScale2]));
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: attentionBuffer } },
        { binding: 1, resource: { buffer: weightsBuffer } },
        { binding: 2, resource: { buffer: scalesBuffer } },
        { binding: 3, resource: { buffer: hiddenBuffer } },
        { binding: 4, resource: { buffer: normWeightsBuffer } },
        { binding: 5, resource: { buffer: partialBuffer } },
        { binding: 6, resource: { buffer: ffnInputBuffer } },
        { binding: 7, resource: { buffer: ffnInputSumBuffer } },
        { binding: 8, resource: { buffer: paramsBuffer } },
      ],
    }),
    hiddenBuffer,
    ffnInputBuffer,
    ffnInputSumBuffer,
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

async function resetAndDispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  resources: Resources,
  hidden: Float32Array,
): Promise<void> {
  device.queue.writeBuffer(resources.hiddenBuffer, 0, hidden);
  await dispatch(device, pipeline, resources.bindGroup);
}

async function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "DecodeOprojNorm dispatch" });
  const pass = encoder.beginComputePass({ label: "DecodeOprojNorm" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(WORKGROUP_COUNT, 1, 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
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
  return Number(value.toFixed(6));
}