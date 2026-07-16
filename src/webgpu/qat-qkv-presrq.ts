import {
  loadCapturedQatQkvFixture,
  type CapturedQatQkvFixture,
} from "../model/qat-linear-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 32;
const ROWS_PER_WORKGROUP = 2;
const Q_OUT = 2048;
const KV_OUT = 256;
const Q_WORKGROUPS = Q_OUT / ROWS_PER_WORKGROUP;
const KV_WORKGROUPS = KV_OUT / ROWS_PER_WORKGROUP;
const WORKGROUP_COUNT = Q_WORKGROUPS + 2 * KV_WORKGROUPS;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface PresrqQatQkvBenchmarkResult {
  sourceOperator: "com.xenova.gemma4.DecodeQkvProj";
  sourceVariant: "presrq";
  implementation: "combined-storage" | "source-layout";
  artifactSha256: string;
  referenceArtifactSha256: string;
  inFeatures: number;
  qOutFeatures: number;
  kvOutFeatures: number;
  workgroupSize: number;
  rowsPerWorkgroup: number;
  qWorkgroupCount: number;
  kvWorkgroupCount: number;
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
  qMaximumAbsoluteError: number;
  qMaximumRelativeError: number;
  kMaximumAbsoluteError: number;
  kMaximumRelativeError: number;
  vMaximumAbsoluteError: number;
  vMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  compileMs: number;
}

interface QkvResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();
const sourceLayoutPipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkCapturedQatQkvPresrq(
  iterations = 20,
): Promise<PresrqQatQkvBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const fixture = await loadCapturedQatQkvFixture();
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

    const qBytes = Q_OUT * 4;
    const kvBytes = KV_OUT * 4;
    const encoder = device.createCommandEncoder({ label: "DecodeQkvProj readback" });
    encoder.copyBufferToBuffer(resources.outputBuffer, 0, resources.readBuffer, 0, qBytes + 2 * kvBytes);
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const output = resources.readBuffer.getMappedRange();
    const actualQ = new Float32Array(output.slice(0, qBytes));
    const actualK = new Float32Array(output.slice(qBytes, qBytes + kvBytes));
    const actualV = new Float32Array(output.slice(qBytes + kvBytes));
    resources.readBuffer.unmap();

    const qErrors = measureErrors(actualQ, fixture.expectedQ);
    const kErrors = measureErrors(actualK, fixture.expectedK);
    const vErrors = measureErrors(actualV, fixture.expectedV);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(device, compiled.pipeline, resources.bindGroup);

    return {
      sourceOperator: "com.xenova.gemma4.DecodeQkvProj",
      sourceVariant: "presrq",
      implementation: "combined-storage",
      artifactSha256: fixture.artifactSha256,
      referenceArtifactSha256: fixture.referenceSha256,
      inFeatures: 1536,
      qOutFeatures: Q_OUT,
      kvOutFeatures: KV_OUT,
      workgroupSize: WORKGROUP_SIZE,
      rowsPerWorkgroup: ROWS_PER_WORKGROUP,
      qWorkgroupCount: Q_WORKGROUPS,
      kvWorkgroupCount: KV_WORKGROUPS,
      workgroupCount: WORKGROUP_COUNT,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      qMaximumAbsoluteError: qErrors.maximumAbsoluteError,
      qMaximumRelativeError: qErrors.maximumRelativeError,
      kMaximumAbsoluteError: kErrors.maximumAbsoluteError,
      kMaximumRelativeError: kErrors.maximumRelativeError,
      vMaximumAbsoluteError: vErrors.maximumAbsoluteError,
      vMaximumRelativeError: vErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

export async function benchmarkCapturedQatQkvSourceLayout(
  iterations = 20,
): Promise<PresrqQatQkvBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const fixture = await loadCapturedQatQkvFixture();
  const device = await getWebGpuDevice();
  if (device.limits.maxStorageBuffersPerShaderStage < 9) {
    throw new Error("DecodeQkvProj source layout requires nine storage bindings");
  }
  const subgroupReduction = device.features.has("subgroups");
  const cached = sourceLayoutPipelineCache.get(device);
  const compiledPromise = cached ?? compileSourceLayoutPipeline(device, subgroupReduction);
  if (!cached) sourceLayoutPipelineCache.set(device, compiledPromise);

  let compiled: CompiledPipeline;
  try {
    compiled = await compiledPromise;
  } catch (error) {
    sourceLayoutPipelineCache.delete(device);
    throw error;
  }

  const resources = createSourceLayoutResources(device, compiled.pipeline, fixture);
  try {
    await dispatch(device, compiled.pipeline, resources.bindGroup);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, compiled.pipeline, resources.bindGroup);
      dispatchSamples.push(performance.now() - started);
    }

    const qBytes = Q_OUT * 4;
    const kvBytes = KV_OUT * 4;
    const encoder = device.createCommandEncoder({ label: "DecodeQkvProj source layout readback" });
    encoder.copyBufferToBuffer(resources.qOutputBuffer, 0, resources.readBuffer, 0, qBytes);
    encoder.copyBufferToBuffer(resources.kOutputBuffer, 0, resources.readBuffer, qBytes, kvBytes);
    encoder.copyBufferToBuffer(
      resources.vOutputBuffer,
      0,
      resources.readBuffer,
      qBytes + kvBytes,
      kvBytes,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const output = resources.readBuffer.getMappedRange();
    const actualQ = new Float32Array(output.slice(0, qBytes));
    const actualK = new Float32Array(output.slice(qBytes, qBytes + kvBytes));
    const actualV = new Float32Array(output.slice(qBytes + kvBytes));
    resources.readBuffer.unmap();

    const qErrors = measureErrors(actualQ, fixture.expectedQ);
    const kErrors = measureErrors(actualK, fixture.expectedK);
    const vErrors = measureErrors(actualV, fixture.expectedV);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(device, compiled.pipeline, resources.bindGroup);

    return {
      sourceOperator: "com.xenova.gemma4.DecodeQkvProj",
      sourceVariant: "presrq",
      implementation: "source-layout",
      artifactSha256: fixture.artifactSha256,
      referenceArtifactSha256: fixture.referenceSha256,
      inFeatures: 1536,
      qOutFeatures: Q_OUT,
      kvOutFeatures: KV_OUT,
      workgroupSize: WORKGROUP_SIZE,
      rowsPerWorkgroup: ROWS_PER_WORKGROUP,
      qWorkgroupCount: Q_WORKGROUPS,
      kvWorkgroupCount: KV_WORKGROUPS,
      workgroupCount: WORKGROUP_COUNT,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      qMaximumAbsoluteError: qErrors.maximumAbsoluteError,
      qMaximumRelativeError: qErrors.maximumRelativeError,
      kMaximumAbsoluteError: kErrors.maximumAbsoluteError,
      kMaximumRelativeError: kErrors.maximumRelativeError,
      vMaximumAbsoluteError: vErrors.maximumAbsoluteError,
      vMaximumRelativeError: vErrors.maximumRelativeError,
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
  const module = device.createShaderModule({ code: createDecodeQkvPresrqShader(subgroupReduction) });
  const pipeline = await device.createComputePipelineAsync({
    label: "DecodeQkvProj presrq",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

async function compileSourceLayoutPipeline(
  device: GPUDevice,
  subgroupReduction: boolean,
): Promise<CompiledPipeline> {
  const started = performance.now();
  const module = device.createShaderModule({
    code: createDecodeQkvSourceLayoutShader(subgroupReduction),
  });
  const pipeline = await device.createComputePipelineAsync({
    label: "DecodeQkvProj source layout",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

export function createDecodeQkvPresrqShader(
  subgroupReduction: boolean,
  qOutFeatures: 2048 | 4096 = Q_OUT,
  kvOutFeatures: 256 | 512 = KV_OUT,
): string {
  return createDecodeQkvShader(
    subgroupReduction,
    false,
    qOutFeatures,
    kvOutFeatures,
  );
}

export function createDecodeQkvPresrqCacheShader(
  subgroupReduction: boolean,
  qOutFeatures: 2048 | 4096 = Q_OUT,
  kvOutFeatures: 256 | 512 = KV_OUT,
): string {
  return createDecodeQkvShader(
    subgroupReduction,
    true,
    qOutFeatures,
    kvOutFeatures,
  );
}

export function createDecodeQkvSourceLayoutShader(
  subgroupReduction: boolean,
  qOutFeatures: 2048 | 4096 = Q_OUT,
  kvOutFeatures: 256 | 512 = KV_OUT,
): string {
  const qWorkgroups = qOutFeatures / ROWS_PER_WORKGROUP;
  const kvWorkgroups = kvOutFeatures / ROWS_PER_WORKGROUP;
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
struct Params {
  qOutScale: f32,
  kOutScale: f32,
  vOutScale: f32,
}

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> q_bits: array<u32>;
@group(0) @binding(2) var<storage, read> k_bits: array<u32>;
@group(0) @binding(3) var<storage, read> v_bits: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> sum_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_k: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_v: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

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
  let workgroup = workgroup_id.x;
  let lane = local_id.x;
  if (workgroup < ${qWorkgroups}u) {
    let row_base = workgroup * ${ROWS_PER_WORKGROUP}u;
${createSourceProjectionBody("q_bits", "out_q", qOutFeatures, "0u", "params.qOutScale")}
  } else if (workgroup < ${qWorkgroups + kvWorkgroups}u) {
    let row_base = (workgroup - ${qWorkgroups}u) * ${ROWS_PER_WORKGROUP}u;
${createSourceProjectionBody("k_bits", "out_k", kvOutFeatures, `${qOutFeatures}u`, "params.kOutScale")}
  } else {
    let row_base = (workgroup - ${qWorkgroups + kvWorkgroups}u) * ${ROWS_PER_WORKGROUP}u;
${createSourceProjectionBody("v_bits", "out_v", kvOutFeatures, `${qOutFeatures + kvOutFeatures}u`, "params.vOutScale")}
  }
}`;
}

function createSourceProjectionBody(
  bitsName: "q_bits" | "k_bits" | "v_bits",
  outputName: "out_q" | "out_k" | "out_v",
  outputFeatures: 2048 | 4096 | 256 | 512,
  scaleOffset: string,
  outputScale: string,
): string {
  return `    var sum_qa: array<f32, ${ROWS_PER_WORKGROUP}>;
    for (var row = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
      sum_qa[row] = 0.0;
    }
    var word = lane;
    loop {
      if (word >= WORDS_PER_ROW) { break; }
      let activation0 = a[word * 2u];
      let activation1 = a[word * 2u + 1u];
      for (var row = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
        let output_row = row_base + row;
        if (output_row < ${outputFeatures}u) {
          let packed = ${bitsName}[output_row * WORDS_PER_ROW + word];
          let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
          sum_qa[row] = sum_qa[row] +
            dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), activation0) +
            dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), activation1);
        }
      }
      word = word + ${WORKGROUP_SIZE}u;
    }

    let activation_sum = sum_a[0];
    for (var row = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
      let reduced = reduce_sum(sum_qa[row], lane);
      let output_row = row_base + row;
      if (lane == 0u && output_row < ${outputFeatures}u) {
        ${outputName}[output_row] = srq(
          scales[${scaleOffset} + output_row] * (reduced - ZERO_POINT * activation_sum),
          ${outputScale},
        );
      }
    }`;
}

function createDecodeQkvShader(
  subgroupReduction: boolean,
  directValueCache: boolean,
  qOutFeatures: 2048 | 4096,
  kvOutFeatures: 256 | 512,
): string {
  const qWorkgroups = qOutFeatures / ROWS_PER_WORKGROUP;
  const kvWorkgroups = kvOutFeatures / ROWS_PER_WORKGROUP;
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

  const params = directValueCache
    ? `struct Params {
  qOutScale: f32,
  kOutScale: f32,
  vOutScale: f32,
  vCacheOffset: u32,
}`
    : `struct Params {
  qOutScale: f32,
  kOutScale: f32,
  vOutScale: f32,
}`;
  const valueCacheBinding = directValueCache
    ? "@group(0) @binding(6) var<storage, read_write> valueCache: array<f32>;"
    : "";

  return `${reduction}
${params}

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> sum_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
${valueCacheBinding}

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
  let workgroup = workgroup_id.x;
  let lane = local_id.x;
  var row_base = 0u;
  var row_limit = 0u;
  var output_scale = 0.0;
  if (workgroup < ${qWorkgroups}u) {
    row_base = workgroup * ${ROWS_PER_WORKGROUP}u;
    row_limit = ${qOutFeatures}u;
    output_scale = params.qOutScale;
  } else if (workgroup < ${qWorkgroups + kvWorkgroups}u) {
    row_base = ${qOutFeatures}u + (workgroup - ${qWorkgroups}u) * ${ROWS_PER_WORKGROUP}u;
    row_limit = ${qOutFeatures + kvOutFeatures}u;
    output_scale = params.kOutScale;
  } else {
    row_base = ${qOutFeatures + kvOutFeatures}u + (workgroup - ${qWorkgroups + kvWorkgroups}u) * ${ROWS_PER_WORKGROUP}u;
    row_limit = ${qOutFeatures + 2 * kvOutFeatures}u;
    output_scale = params.vOutScale;
  }
${createProjectionBody(directValueCache, qOutFeatures, kvOutFeatures)}
}`;
}

function createProjectionBody(
  directValueCache: boolean,
  qOutFeatures: 2048 | 4096,
  kvOutFeatures: 256 | 512,
): string {
  const outputWrite = directValueCache
    ? `        let projected = srq(
          scales[output_row] * (reduced - ZERO_POINT * activation_sum),
          output_scale,
        );
        if (output_row < ${qOutFeatures + kvOutFeatures}u) {
          output[output_row] = projected;
        } else {
          valueCache[params.vCacheOffset + output_row - ${qOutFeatures + kvOutFeatures}u] = projected;
        }`
    : `        output[output_row] = srq(
          scales[output_row] * (reduced - ZERO_POINT * activation_sum),
          output_scale,
        );`;
  return `    var sum_qa: array<f32, ${ROWS_PER_WORKGROUP}>;
    for (var row: u32 = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
      sum_qa[row] = 0.0;
    }
    var word = lane;
    loop {
      if (word >= WORDS_PER_ROW) { break; }
      var activation_chunk: array<vec4<f32>, 2>;
      for (var chunk: u32 = 0u; chunk < 2u; chunk = chunk + 1u) {
        activation_chunk[chunk] = a[word * 2u + chunk];
      }
      for (var row: u32 = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
        let output_row = row_base + row;
        if (output_row < row_limit) {
          let packed = bits[output_row * WORDS_PER_ROW + word];
          let lo = vec4<f32>(unpack4xU8(packed & 0x0f0f0f0fu));
          let hi = vec4<f32>(unpack4xU8((packed >> 4u) & 0x0f0f0f0fu));
          sum_qa[row] = sum_qa[row] +
            dot(vec4<f32>(lo.x, hi.x, lo.y, hi.y), activation_chunk[0]) +
            dot(vec4<f32>(lo.z, hi.z, lo.w, hi.w), activation_chunk[1]);
        }
      }
      word = word + ${WORKGROUP_SIZE}u;
    }

    let activation_sum = sum_a[0];
    for (var row: u32 = 0u; row < ${ROWS_PER_WORKGROUP}u; row = row + 1u) {
      let reduced = reduce_sum(sum_qa[row], lane);
      let output_row = row_base + row;
      if (lane == 0u && output_row < row_limit) {
${outputWrite}
      }
    }`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: CapturedQatQkvFixture,
): QkvResources {
  const inputBuffer = createBuffer(device, "QKV presrq activation", fixture.input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "QKV packed weights", fixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scaleBuffer = createBuffer(device, "QKV row scales", fixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sumBuffer = createBuffer(device, "QKV activation sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "QKV output", (Q_OUT + 2 * KV_OUT) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuffer = createBuffer(device, "QKV readback", (Q_OUT + 2 * KV_OUT) * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramsBuffer = createBuffer(device, "QKV parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const buffers = [inputBuffer, weightBuffer, scaleBuffer, sumBuffer, outputBuffer, readBuffer, paramsBuffer];

  device.queue.writeBuffer(inputBuffer, 0, fixture.input);
  device.queue.writeBuffer(weightBuffer, 0, fixture.packedWeights);
  device.queue.writeBuffer(scaleBuffer, 0, fixture.rowScales);
  device.queue.writeBuffer(sumBuffer, 0, fixture.inputSum);
  device.queue.writeBuffer(paramsBuffer, 0, fixture.outputScales);

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

interface SourceLayoutResources {
  bindGroup: GPUBindGroup;
  qOutputBuffer: GPUBuffer;
  kOutputBuffer: GPUBuffer;
  vOutputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

function createSourceLayoutResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: CapturedQatQkvFixture,
): SourceLayoutResources {
  const qWeightBytes = Q_OUT * 192 * Uint32Array.BYTES_PER_ELEMENT;
  const kvWeightBytes = KV_OUT * 192 * Uint32Array.BYTES_PER_ELEMENT;
  const inputBuffer = createBuffer(device, "QKV source activation", fixture.input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const weightBuffer = createBuffer(device, "QKV source packed weights", fixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const scaleBuffer = createBuffer(device, "QKV source row scales", fixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const sumBuffer = createBuffer(device, "QKV source activation sum", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qOutputBuffer = createBuffer(device, "QKV source Q output", Q_OUT * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const kOutputBuffer = createBuffer(device, "QKV source K output", KV_OUT * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const vOutputBuffer = createBuffer(device, "QKV source V output", KV_OUT * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuffer = createBuffer(device, "QKV source readback", (Q_OUT + 2 * KV_OUT) * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const paramsBuffer = createBuffer(device, "QKV source parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const buffers = [
    inputBuffer,
    weightBuffer,
    scaleBuffer,
    sumBuffer,
    qOutputBuffer,
    kOutputBuffer,
    vOutputBuffer,
    readBuffer,
    paramsBuffer,
  ];

  device.queue.writeBuffer(inputBuffer, 0, fixture.input);
  device.queue.writeBuffer(weightBuffer, 0, fixture.packedWeights);
  device.queue.writeBuffer(scaleBuffer, 0, fixture.rowScales);
  device.queue.writeBuffer(sumBuffer, 0, fixture.inputSum);
  device.queue.writeBuffer(paramsBuffer, 0, fixture.outputScales);

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer, size: qWeightBytes } },
        { binding: 2, resource: { buffer: weightBuffer, offset: qWeightBytes, size: kvWeightBytes } },
        { binding: 3, resource: { buffer: weightBuffer, offset: qWeightBytes + kvWeightBytes, size: kvWeightBytes } },
        { binding: 4, resource: { buffer: scaleBuffer } },
        { binding: 5, resource: { buffer: sumBuffer } },
        { binding: 6, resource: { buffer: qOutputBuffer } },
        { binding: 7, resource: { buffer: kOutputBuffer } },
        { binding: 8, resource: { buffer: vOutputBuffer } },
        { binding: 9, resource: { buffer: paramsBuffer } },
      ],
    }),
    qOutputBuffer,
    kOutputBuffer,
    vOutputBuffer,
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
  const encoder = device.createCommandEncoder({ label: "DecodeQkvProj dispatch" });
  const pass = encoder.beginComputePass({ label: "DecodeQkvProj presrq" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(WORKGROUP_COUNT);
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
    label: "QKV timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "QKV timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "QKV timestamp sample" });
      const pass = encoder.beginComputePass({
        label: "DecodeQkvProj timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < DISPATCHES_PER_TIMESTAMP_SAMPLE; index += 1) {
        pass.dispatchWorkgroups(WORKGROUP_COUNT);
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