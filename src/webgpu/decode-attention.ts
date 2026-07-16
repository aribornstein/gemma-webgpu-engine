import {
  loadDecodeAttentionFixture,
  type DecodeAttentionFixture,
} from "../model/decode-attention-fixture";
import { getWebGpuDevice } from "./device";
import { DecodeKvCache } from "./decode-kv-cache";

const WORKGROUP_SIZE = 256;
const CHUNK_COUNT = 32;
const WORKGROUP_COUNT = 8 * CHUNK_COUNT;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 20;
const PARTIAL_UINT32_ELEMENTS = 8 * CHUNK_COUNT * (256 + 2) + 8;

export interface DecodeAttentionBenchmarkResult {
  sourceOperator: "Gemma4DecodeAttentionPartial";
  sourceVariant: "fixed-subgroup-32";
  artifactSha256: string;
  sourceCaptureSha256: string;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  keyLength: number;
  queryOffset: number;
  window: number;
  workgroupSize: number;
  chunkCount: number;
  workgroupCount: number;
  iterations: number;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  gpuKernelDispatchesPerSample: number | null;
  gpuKernelSamplesMs: number[] | null;
  gpuKernelMedianMs: number | null;
  gpuKernelP95Ms: number | null;
  outputMaximumAbsoluteError: number;
  outputMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  compileMs: number;
}

interface DecodeAttentionResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  cache: DecodeKvCache;
  buffers: GPUBuffer[];
  bufferCount: number;
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkDecodeAttention(
  iterations = 20,
): Promise<DecodeAttentionBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }
  const fixture = await loadDecodeAttentionFixture();
  const device = await getWebGpuDevice();
  if (!device.features.has("subgroups")) {
    throw new Error("Decode attention requires WebGPU subgroups");
  }
  const cached = pipelineCache.get(device);
  const compiledPromise = cached ?? compilePipeline(device);
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

    const encoder = device.createCommandEncoder({ label: "Decode attention readback" });
    encoder.copyBufferToBuffer(
      resources.outputBuffer,
      0,
      resources.readBuffer,
      0,
      fixture.expectedOutput.byteLength,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const actualOutput = new Float32Array(resources.readBuffer.getMappedRange().slice(0));
    resources.readBuffer.unmap();

    const errors = measureErrors(actualOutput, fixture.expectedOutput);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(device, compiled.pipeline, resources.bindGroup);
    return {
      sourceOperator: "Gemma4DecodeAttentionPartial",
      sourceVariant: "fixed-subgroup-32",
      artifactSha256: fixture.artifactSha256,
      sourceCaptureSha256: fixture.sourceCaptureSha256,
      qHeads: fixture.qHeads,
      kvHeads: fixture.kvHeads,
      headDim: fixture.headDim,
      keyLength: fixture.keyLength,
      queryOffset: fixture.queryOffset,
      window: fixture.window,
      workgroupSize: WORKGROUP_SIZE,
      chunkCount: CHUNK_COUNT,
      workgroupCount: WORKGROUP_COUNT,
      iterations,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelSamplesMs: gpuSamples?.map(round) ?? null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      outputMaximumAbsoluteError: errors.maximumAbsoluteError,
      outputMaximumRelativeError: errors.maximumRelativeError,
      gpuBufferAllocations: resources.bufferCount,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
    resources.cache.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<CompiledPipeline> {
  const started = performance.now();
  const module = device.createShaderModule({ code: createDecodeAttentionShader() });
  const pipeline = await device.createComputePipelineAsync({
    label: "Gemma4DecodeAttentionPartial fixed subgroup 32",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

export function createDecodeAttentionShader(
  headDim: 256 | 512 = 256,
): string {
  const halfDim = headDim / 2;
  const vec4Dimensions = headDim / 4;
  const valueGroups = 256 / vec4Dimensions;
  const partialCounterBase = 8 * 32 * (headDim + 2);
  return `enable subgroups;

struct Params {
  seqQ: u32,
  keyLen: u32,
  qOffset: u32,
  qHeads: u32,
  kvHeads: u32,
  window: u32,
  outQuantScale: f32,
  cacheCapacity: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> cosTbl: array<f32>;
@group(0) @binding(3) var<storage, read> sinTbl: array<f32>;
@group(0) @binding(4) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;

const HEAD_DIM: u32 = ${headDim}u;
const HALF_DIM: u32 = ${halfDim}u;
const NCHUNK: u32 = 32u;
const WG: u32 = 256u;
const EPS: f32 = 0.000001;
const SCALE: f32 = 1.0;
const NEG_INF: f32 = -3.4028234663852886e38;
const PP_COUNTER_BASE: u32 = ${partialCounterBase}u;

var<workgroup> lastFlag: u32;
var<workgroup> qn_sh: array<f32, ${headDim}>;
var<workgroup> out_acc: array<f32, ${headDim}>;
var<workgroup> probs: array<f32, 256>;
var<workgroup> sval_sh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> wgt_sh: array<f32, 32>;
var<workgroup> vacc_sh: array<vec4<f32>, 256>;
var<workgroup> running_max: f32;
var<workgroup> running_denom: f32;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn subgroup_sum(value: f32) -> f32 {
  return subgroupAdd(value);
}

fn subgroup_max(value: f32) -> f32 {
  return subgroupMax(value);
}

fn reduce_max(value: f32, thread: u32) -> f32 {
  let subgroup_value = subgroup_max(value);
  if ((thread & 31u) == 0u) { red[thread >> 5u] = subgroup_value; }
  workgroupBarrier();
  var total = NEG_INF;
  for (var index = 0u; index < 8u; index = index + 1u) {
    total = max(total, red[index]);
  }
  workgroupBarrier();
  return total;
}

fn reduce_sum(value: f32, thread: u32) -> f32 {
  let subgroup_value = subgroup_sum(value);
  if ((thread & 31u) == 0u) { red[thread >> 5u] = subgroup_value; }
  workgroupBarrier();
  var total = 0.0;
  for (var index = 0u; index < 8u; index = index + 1u) {
    total = total + red[index];
  }
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let head = workgroup_id.x;
  let chunk = workgroup_id.y;
  if (head >= params.qHeads) { return; }

  let thread = local_id.x;
  let group_size = params.qHeads / params.kvHeads;
  let kv_head = head / group_size;
  let query_position = params.qOffset;
  let query_base = head * HEAD_DIM;
  let maximum_key = min(params.keyLen, query_position + 1u);
  var minimum_key = 0u;
  if (params.window > 0u && query_position + 1u > params.window) {
    minimum_key = query_position + 1u - params.window;
  }
  let active_keys = maximum_key - minimum_key;
  let active_chunks = clamp((active_keys + 63u) / 64u, 8u, NCHUNK);
  if (chunk >= active_chunks) { return; }

  var square_sum = 0.0;
  var dimension = thread;
  loop {
    if (dimension >= HEAD_DIM) { break; }
    let value = q[query_base + dimension];
    square_sum = square_sum + value * value;
    dimension = dimension + WG;
  }
  let norm_scale = inverseSqrt(reduce_sum(square_sum, thread) / f32(HEAD_DIM) + EPS);

  var pair = thread;
  loop {
    if (pair >= HALF_DIM) { break; }
    let first = q[query_base + pair] * norm_scale * w[pair];
    let second = q[query_base + pair + HALF_DIM] * norm_scale * w[pair + HALF_DIM];
    let cosine = cosTbl[pair];
    let sine = sinTbl[pair];
    qn_sh[pair] = first * cosine - second * sine;
    qn_sh[pair + HALF_DIM] = second * cosine + first * sine;
    pair = pair + WG;
  }
  for (var index = thread; index < HEAD_DIM; index = index + WG) {
    out_acc[index] = 0.0;
  }
  if (thread == 0u) {
    running_max = NEG_INF;
    running_denom = 0.0;
  }
  workgroupBarrier();

  let chunk_length = (active_keys + active_chunks - 1u) / active_chunks;
  let start = minimum_key + chunk * chunk_length;
  let end = min(start + chunk_length, maximum_key);
  var tile = start;
  loop {
    if (tile >= end) { break; }
    let key_index = tile + thread;
    let tile_count_for_scores = min(WG, end - tile);
    let subgroup_rounds = (tile_count_for_scores + 7u) / 8u;
    for (var round = 0u; round < subgroup_rounds; round = round + 1u) {
      let key_in_tile = round * 8u + (thread / 32u);
      var score_accumulator = 0.0;
      if (key_in_tile < tile_count_for_scores) {
        let physical_key = (tile + key_in_tile) % params.cacheCapacity;
        let key_base = (physical_key * params.kvHeads + kv_head) * ${vec4Dimensions}u;
        for (var dimension4 = thread & 31u; dimension4 < ${vec4Dimensions}u; dimension4 = dimension4 + 32u) {
          let key4 = k[key_base + dimension4];
          score_accumulator = score_accumulator + dot(
            vec4<f32>(
              qn_sh[dimension4 * 4u],
              qn_sh[dimension4 * 4u + 1u],
              qn_sh[dimension4 * 4u + 2u],
              qn_sh[dimension4 * 4u + 3u]
            ),
            key4,
          );
        }
      }
      let score = subgroup_sum(score_accumulator);
      if ((thread & 31u) == 0u && key_in_tile < tile_count_for_scores) {
        sval_sh[key_in_tile] = score * SCALE;
      }
    }
    workgroupBarrier();
    var scaled_score = NEG_INF;
    if (key_index < end) { scaled_score = sval_sh[thread]; }

    let tile_maximum = reduce_max(scaled_score, thread);
    let new_maximum = max(running_max, tile_maximum);
    let correction = exp(running_max - new_maximum);
    var probability = 0.0;
    if (key_index < end) { probability = exp(scaled_score - new_maximum); }
    probs[thread] = probability;
    let tile_denominator = reduce_sum(probability, thread);
    if (thread == 0u) {
      running_denom = running_denom * correction + tile_denominator;
      running_max = new_maximum;
    }
    workgroupBarrier();

    let tile_count = min(WG, end - tile);
    let key_group = thread / ${vec4Dimensions}u;
    let value_dimension4 = thread % ${vec4Dimensions}u;
    var value_accumulator = vec4<f32>(0.0);
    var key_offset = key_group;
    loop {
      if (key_offset >= tile_count) { break; }
      let physical_key = (tile + key_offset) % params.cacheCapacity;
      let value_base = (physical_key * params.kvHeads + kv_head) * ${vec4Dimensions}u;
      value_accumulator = value_accumulator +
        probs[key_offset] * v[value_base + value_dimension4];
      key_offset = key_offset + ${valueGroups}u;
    }
    vacc_sh[thread] = value_accumulator;
    workgroupBarrier();
    for (var dimension4 = thread; dimension4 < ${vec4Dimensions}u; dimension4 = dimension4 + WG) {
      var accumulated4 = vec4<f32>(
        out_acc[dimension4 * 4u],
        out_acc[dimension4 * 4u + 1u],
        out_acc[dimension4 * 4u + 2u],
        out_acc[dimension4 * 4u + 3u]
      ) * correction;
      for (var group = 0u; group < ${valueGroups}u; group = group + 1u) {
        accumulated4 = accumulated4 + vacc_sh[group * ${vec4Dimensions}u + dimension4];
      }
      out_acc[dimension4 * 4u] = accumulated4.x;
      out_acc[dimension4 * 4u + 1u] = accumulated4.y;
      out_acc[dimension4 * 4u + 2u] = accumulated4.z;
      out_acc[dimension4 * 4u + 3u] = accumulated4.w;
    }
    workgroupBarrier();
    tile = tile + WG;
  }

  let partial_base = (head * NCHUNK + chunk) * (HEAD_DIM + 2u);
  for (var index = thread; index < HEAD_DIM; index = index + WG) {
    atomicStore(&partials[partial_base + index], bitcast<u32>(out_acc[index]));
  }
  if (thread == 0u) {
    atomicStore(&partials[partial_base + HEAD_DIM], bitcast<u32>(running_max));
    atomicStore(&partials[partial_base + HEAD_DIM + 1u], bitcast<u32>(running_denom));
  }
  storageBarrier();

  if (thread == 0u) {
    let ticket = atomicAdd(&partials[PP_COUNTER_BASE + head], 1u);
    lastFlag = select(0u, 1u, ticket == active_chunks - 1u);
  }
  if (workgroupUniformLoad(&lastFlag) != 1u) { return; }
  if (thread == 0u) { atomicStore(&partials[PP_COUNTER_BASE + head], 0u); }

  var local_maximum = NEG_INF;
  var local_denominator = 0.0;
  if (thread < active_chunks) {
    let base = (head * NCHUNK + thread) * (HEAD_DIM + 2u);
    local_maximum = bitcast<f32>(atomicLoad(&partials[base + HEAD_DIM]));
    local_denominator = bitcast<f32>(atomicLoad(&partials[base + HEAD_DIM + 1u]));
  }
  let merged_maximum = reduce_max(local_maximum, thread);
  var local_weight = 0.0;
  if (thread < active_chunks) {
    local_weight = exp(local_maximum - merged_maximum);
    wgt_sh[thread] = local_weight;
  }
  let denominator = reduce_sum(local_denominator * local_weight, thread);
  let inverse_denominator = 1.0 / denominator;
  for (var output_dimension = thread; output_dimension < HEAD_DIM; output_dimension = output_dimension + WG) {
    var accumulator = 0.0;
    for (var source_chunk = 0u; source_chunk < active_chunks; source_chunk = source_chunk + 1u) {
      accumulator = accumulator + bitcast<f32>(atomicLoad(
        &partials[(head * NCHUNK + source_chunk) * (HEAD_DIM + 2u) + output_dimension]
      )) * wgt_sh[source_chunk];
    }
    out[head * HEAD_DIM + output_dimension] = srq(
      accumulator * inverse_denominator,
      params.outQuantScale,
    );
  }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: DecodeAttentionFixture,
): DecodeAttentionResources {
  const qBuffer = createBuffer(device, "Decode attention Q", fixture.q.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "Decode attention Q norm weight", fixture.qNormWeight.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const cosineBuffer = createBuffer(device, "Decode attention cosine", fixture.cosine.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sineBuffer = createBuffer(device, "Decode attention sine", fixture.sine.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const cache = new DecodeKvCache(device, {
    capacity: fixture.keyLength,
    kvHeads: fixture.kvHeads,
    headDim: fixture.headDim,
    label: "Decode attention cache",
  });
  const partialBuffer = createBuffer(device, "Decode attention partials", PARTIAL_UINT32_ELEMENTS * 4, GPUBufferUsage.STORAGE);
  const outputBuffer = createBuffer(device, "Decode attention output", fixture.expectedOutput.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const paramsBuffer = createBuffer(device, "Decode attention parameters", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readBuffer = createBuffer(device, "Decode attention readback", fixture.expectedOutput.byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [qBuffer, weightBuffer, cosineBuffer, sineBuffer, partialBuffer, outputBuffer, paramsBuffer, readBuffer];

  device.queue.writeBuffer(qBuffer, 0, fixture.q);
  device.queue.writeBuffer(weightBuffer, 0, fixture.qNormWeight);
  device.queue.writeBuffer(cosineBuffer, 0, fixture.cosine);
  device.queue.writeBuffer(sineBuffer, 0, fixture.sine);
  cache.writeTokens(device.queue, 0, fixture.keyCache, fixture.valueCache);
  const params = new ArrayBuffer(32);
  const paramsView = new DataView(params);
  paramsView.setUint32(0, 1, true);
  paramsView.setUint32(4, fixture.keyLength, true);
  paramsView.setUint32(8, fixture.queryOffset, true);
  paramsView.setUint32(12, fixture.qHeads, true);
  paramsView.setUint32(16, fixture.kvHeads, true);
  paramsView.setUint32(20, fixture.window, true);
  paramsView.setFloat32(24, fixture.outputQuantScale, true);
  paramsView.setUint32(28, fixture.keyLength, true);
  device.queue.writeBuffer(paramsBuffer, 0, params);

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: cosineBuffer } },
        { binding: 3, resource: { buffer: sineBuffer } },
        { binding: 4, resource: { buffer: cache.keyBuffer } },
        { binding: 5, resource: { buffer: cache.valueBuffer } },
        { binding: 6, resource: { buffer: partialBuffer } },
        { binding: 7, resource: { buffer: outputBuffer } },
        { binding: 8, resource: { buffer: paramsBuffer } },
      ],
    }),
    outputBuffer,
    readBuffer,
    cache,
    buffers,
    bufferCount: buffers.length + cache.buffers.length,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0) + cache.bytesAllocated,
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
  const encoder = device.createCommandEncoder({ label: "Decode attention dispatch" });
  const pass = encoder.beginComputePass({ label: "Gemma4DecodeAttentionPartial" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(8, CHUNK_COUNT, 1);
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
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: "Decode attention timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "Decode attention timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];
  try {
    for (let sample = -2; sample < 10; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "Decode attention timestamp sample" });
      const pass = encoder.beginComputePass({
        label: "Decode attention timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < DISPATCHES_PER_TIMESTAMP_SAMPLE; index += 1) {
        pass.dispatchWorkgroups(8, CHUNK_COUNT, 1);
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
