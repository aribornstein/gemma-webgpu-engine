import { loadDecodeOprojNormFixture } from "../model/decode-oproj-norm-fixture";
import { getWebGpuDevice } from "./device";
import { measureGpuDispatches } from "./gpu-timestamp";

const WORKGROUP_SIZE = 256;
const OUT_FEATURES = 1536;
const WORKGROUP_COUNT = 192;

export type DecodeOprojNormMode = "subgroup-rows" | "cooperative-rows";

export interface DecodeOprojNormBenchmarkResult {
  sourceOperator: "com.xenova.gemma4.DecodeOprojNorm";
  sourceVariant: "fused-fixed-subgroup-32" | "fused-row-cooperative-32";
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
  gpuKernelDispatchesPerSample: number | null;
  gpuKernelSamplesMs: number[] | null;
  gpuKernelMedianMs: number | null;
  gpuKernelP95Ms: number | null;
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

const pipelineCache = new WeakMap<GPUDevice, Map<DecodeOprojNormMode, Promise<CompiledPipeline>>>();

export async function benchmarkDecodeOprojNorm(
  iterations = 20,
  mode: DecodeOprojNormMode = "subgroup-rows",
  gpuDispatchesPerSample = 20,
): Promise<DecodeOprojNormBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }
  if (!Number.isInteger(gpuDispatchesPerSample) || gpuDispatchesPerSample < 1) {
    throw new Error("GPU dispatches per sample must be a positive integer");
  }
  const [fixture, device] = await Promise.all([
    loadDecodeOprojNormFixture(),
    getWebGpuDevice(),
  ]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("DecodeOprojNorm requires WebGPU subgroups and shader-f16");
  }
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const compiledPromise = devicePipelines.get(mode) ?? compilePipeline(device, mode);
  if (!devicePipelines.has(mode)) devicePipelines.set(mode, compiledPromise);
  let compiled: CompiledPipeline;
  try {
    compiled = await compiledPromise;
  } catch (error) {
    devicePipelines.delete(mode);
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
    const gpuSamples = await measureGpuDispatches(
      device,
      "DecodeOprojNorm",
      gpuDispatchesPerSample,
      (pass) => {
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, resources.bindGroup);
        pass.dispatchWorkgroups(WORKGROUP_COUNT);
      },
      10,
      () => device.queue.writeBuffer(resources.hiddenBuffer, 0, fixture.hiddenBefore),
    );
    const sortedSamples = samples.toSorted((left, right) => left - right);
    return {
      sourceOperator: "com.xenova.gemma4.DecodeOprojNorm",
      sourceVariant: mode === "cooperative-rows"
        ? "fused-row-cooperative-32"
        : "fused-fixed-subgroup-32",
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
      gpuKernelDispatchesPerSample: gpuSamples ? gpuDispatchesPerSample : null,
      gpuKernelSamplesMs: gpuSamples?.map(round) ?? null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipeline(
  device: GPUDevice,
  mode: DecodeOprojNormMode,
): Promise<CompiledPipeline> {
  const module = device.createShaderModule({ code: createDecodeOprojNormShader(2048, mode) });
  const pipeline = await device.createComputePipelineAsync({
    label: `DecodeOprojNorm ${mode}`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline };
}

export function createDecodeOprojNormShader(
  inFeatures: 2048 | 4096 = 2048,
  mode: DecodeOprojNormMode = "subgroup-rows",
): string {
  const wordsPerRow = inFeatures / 8;
  const projectionWorkgroupDeclarations = mode === "cooperative-rows"
    ? `var<workgroup> projectionSums0: array<vec4<f32>, 8>;
var<workgroup> projectionSums1: array<vec4<f32>, 8>;
var<workgroup> projectionActivationSums: array<f32, 8>;`
    : "";
  const projectionBody = mode === "cooperative-rows"
    ? createCooperativeProjectionBody()
    : createSubgroupProjectionBody();
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
${projectionWorkgroupDeclarations}

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
${projectionBody}
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

function createSubgroupProjectionBody(): string {
  return `  let row_base = workgroup_id.x * ROWS_PER_WG + subgroup_id * SG_ROWS;
  var sum_qa = 0.0;
  var sum_a = 0.0;
  var word = lane;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    let activation0 = a[word * 2u];
    let activation1 = a[word * 2u + 1u];
    sum_a = sum_a + activation0.x + activation0.y + activation0.z + activation0.w;
    sum_a = sum_a + activation1.x + activation1.y + activation1.z + activation1.w;
    if (row_base < OUT_F) {
      let packed = bits_buf[row_base * WORDS_PER_ROW + word];
      let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
      let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
      sum_qa = sum_qa +
        dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), activation0) +
        dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), activation1);
    }
    word = word + 32u;
  }
  let reduced_a = sg_sum(sum_a);
  let reduced_qa = sg_sum(sum_qa);
  if (lane == 0u && row_base < OUT_F) {
    atomicStore(&pp[row_base], bitcast<u32>(srq(
      scale[row_base] * (reduced_qa - ZP * reduced_a),
      params.outScale,
    )));
  }`;
}

function createCooperativeProjectionBody(): string {
  return `  let row_base = workgroup_id.x * ROWS_PER_WG;
  var projection0 = 0.0;
  var projection1 = 0.0;
  var projection2 = 0.0;
  var projection3 = 0.0;
  var projection4 = 0.0;
  var projection5 = 0.0;
  var projection6 = 0.0;
  var projection7 = 0.0;
  var sum_a = 0.0;
  var word = tid;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    let activation0 = a[word * 2u];
    let activation1 = a[word * 2u + 1u];
    sum_a = sum_a +
      (activation0.x + activation0.y + activation0.z + activation0.w) +
      (activation1.x + activation1.y + activation1.z + activation1.w);
    let packed0 = bits_buf[(row_base + 0u) * WORDS_PER_ROW + word];
    let packed1 = bits_buf[(row_base + 1u) * WORDS_PER_ROW + word];
    let packed2 = bits_buf[(row_base + 2u) * WORDS_PER_ROW + word];
    let packed3 = bits_buf[(row_base + 3u) * WORDS_PER_ROW + word];
    let packed4 = bits_buf[(row_base + 4u) * WORDS_PER_ROW + word];
    let packed5 = bits_buf[(row_base + 5u) * WORDS_PER_ROW + word];
    let packed6 = bits_buf[(row_base + 6u) * WORDS_PER_ROW + word];
    let packed7 = bits_buf[(row_base + 7u) * WORDS_PER_ROW + word];
    let lo0 = vec4<f32>(unpack4xU8(packed0 & 0x0f0f0f0fu));
    let hi0 = vec4<f32>(unpack4xU8((packed0 >> 4u) & 0x0f0f0f0fu));
    let lo1 = vec4<f32>(unpack4xU8(packed1 & 0x0f0f0f0fu));
    let hi1 = vec4<f32>(unpack4xU8((packed1 >> 4u) & 0x0f0f0f0fu));
    let lo2 = vec4<f32>(unpack4xU8(packed2 & 0x0f0f0f0fu));
    let hi2 = vec4<f32>(unpack4xU8((packed2 >> 4u) & 0x0f0f0f0fu));
    let lo3 = vec4<f32>(unpack4xU8(packed3 & 0x0f0f0f0fu));
    let hi3 = vec4<f32>(unpack4xU8((packed3 >> 4u) & 0x0f0f0f0fu));
    let lo4 = vec4<f32>(unpack4xU8(packed4 & 0x0f0f0f0fu));
    let hi4 = vec4<f32>(unpack4xU8((packed4 >> 4u) & 0x0f0f0f0fu));
    let lo5 = vec4<f32>(unpack4xU8(packed5 & 0x0f0f0f0fu));
    let hi5 = vec4<f32>(unpack4xU8((packed5 >> 4u) & 0x0f0f0f0fu));
    let lo6 = vec4<f32>(unpack4xU8(packed6 & 0x0f0f0f0fu));
    let hi6 = vec4<f32>(unpack4xU8((packed6 >> 4u) & 0x0f0f0f0fu));
    let lo7 = vec4<f32>(unpack4xU8(packed7 & 0x0f0f0f0fu));
    let hi7 = vec4<f32>(unpack4xU8((packed7 >> 4u) & 0x0f0f0f0fu));
    projection0 = projection0 + dot(vec4<f32>(lo0.x, hi0.x, lo0.y, hi0.y), activation0) + dot(vec4<f32>(lo0.z, hi0.z, lo0.w, hi0.w), activation1);
    projection1 = projection1 + dot(vec4<f32>(lo1.x, hi1.x, lo1.y, hi1.y), activation0) + dot(vec4<f32>(lo1.z, hi1.z, lo1.w, hi1.w), activation1);
    projection2 = projection2 + dot(vec4<f32>(lo2.x, hi2.x, lo2.y, hi2.y), activation0) + dot(vec4<f32>(lo2.z, hi2.z, lo2.w, hi2.w), activation1);
    projection3 = projection3 + dot(vec4<f32>(lo3.x, hi3.x, lo3.y, hi3.y), activation0) + dot(vec4<f32>(lo3.z, hi3.z, lo3.w, hi3.w), activation1);
    projection4 = projection4 + dot(vec4<f32>(lo4.x, hi4.x, lo4.y, hi4.y), activation0) + dot(vec4<f32>(lo4.z, hi4.z, lo4.w, hi4.w), activation1);
    projection5 = projection5 + dot(vec4<f32>(lo5.x, hi5.x, lo5.y, hi5.y), activation0) + dot(vec4<f32>(lo5.z, hi5.z, lo5.w, hi5.w), activation1);
    projection6 = projection6 + dot(vec4<f32>(lo6.x, hi6.x, lo6.y, hi6.y), activation0) + dot(vec4<f32>(lo6.z, hi6.z, lo6.w, hi6.w), activation1);
    projection7 = projection7 + dot(vec4<f32>(lo7.x, hi7.x, lo7.y, hi7.y), activation0) + dot(vec4<f32>(lo7.z, hi7.z, lo7.w, hi7.w), activation1);
    word = word + WG;
  }
  let reduced0 = subgroupAdd(vec4<f32>(projection0, projection1, projection2, projection3));
  let reduced1 = subgroupAdd(vec4<f32>(projection4, projection5, projection6, projection7));
  let reduced_a = subgroupAdd(sum_a);
  if (lane == 0u) {
    projectionSums0[subgroup_id] = reduced0;
    projectionSums1[subgroup_id] = reduced1;
    projectionActivationSums[subgroup_id] = reduced_a;
  }
  workgroupBarrier();
  if (tid == 0u) {
    var total0 = vec4<f32>(0.0);
    var total1 = vec4<f32>(0.0);
    var total_a = 0.0;
    for (var index = 0u; index < 8u; index = index + 1u) {
      total0 = total0 + projectionSums0[index];
      total1 = total1 + projectionSums1[index];
      total_a = total_a + projectionActivationSums[index];
    }
    atomicStore(&pp[row_base + 0u], bitcast<u32>(srq(scale[row_base + 0u] * (total0[0] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 1u], bitcast<u32>(srq(scale[row_base + 1u] * (total0[1] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 2u], bitcast<u32>(srq(scale[row_base + 2u] * (total0[2] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 3u], bitcast<u32>(srq(scale[row_base + 3u] * (total0[3] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 4u], bitcast<u32>(srq(scale[row_base + 4u] * (total1[0] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 5u], bitcast<u32>(srq(scale[row_base + 5u] * (total1[1] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 6u], bitcast<u32>(srq(scale[row_base + 6u] * (total1[2] - ZP * total_a), params.outScale)));
    atomicStore(&pp[row_base + 7u], bitcast<u32>(srq(scale[row_base + 7u] * (total1[3] - ZP * total_a), params.outScale)));
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