import { loadDecodeAttentionFixture } from "../model/decode-attention-fixture";
import { loadDecodeKNormRopeFixture } from "../model/decode-k-norm-rope-fixture";
import { loadDecodeOprojNormFixture } from "../model/decode-oproj-norm-fixture";
import { loadDecodeRmsFixture } from "../model/decode-rms-fixture";
import type { MaterializedGemmaLayer } from "../model/gemma-layer-materializer";
import type { GemmaLayerProfile } from "../model/gemma-layer-plan";
import { loadCapturedQatQkvFixture } from "../model/qat-linear-fixture";
import { createDecodeAttentionShader } from "./decode-attention";
import { createDecodeKNormRopeShader } from "./decode-k-norm-rope";
import { DecodeKvCache, resolveDecodeKvCacheAllocation } from "./decode-kv-cache";
import { createDecodeOprojNormShader } from "./decode-oproj-norm";
import { createDecodeRmsSrqShader } from "./decode-rms-srq";
import { createDecodeVRmsShader } from "./decode-v-rms";
import { getWebGpuDevice } from "./device";
import { createDecodeQkvPresrqCacheShader } from "./qat-qkv-presrq";

const Q_OUT = 2048;
const KV_OUT = 256;
const ATTENTION_CHUNK_COUNT = 32;
const OPROJ_WORKGROUP_COUNT = 192;

export interface DecodeAttentionBlockBenchmarkResult {
  sourceOperators: [
    "com.xenova.gemma4.DecodeRmsSrq",
    "com.xenova.gemma4.DecodeQkvProj",
    "com.xenova.gemma4.DecodeQkNormRope",
    "RMSNorm",
    "Gemma4DecodeAttentionPartial",
    "com.xenova.gemma4.DecodeOprojNorm",
  ];
  implementation: "shared-qkv-reusable-kv-cache-oproj";
  cacheCapacity: number;
  cacheLength: number;
  cachePosition: number;
  dispatchesPerToken: 6;
  iterations: number;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  qMaximumAbsoluteError: number;
  qMaximumRelativeError: number;
  rawKMaximumAbsoluteError: number;
  rawKMaximumRelativeError: number;
  cachedKMaximumAbsoluteError: number;
  cachedKMaximumRelativeError: number;
  cachedVMaximumAbsoluteError: number;
  cachedVMaximumRelativeError: number;
  attentionMaximumAbsoluteError: number;
  attentionMaximumRelativeError: number;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  ffnInputBitMismatches: number;
  ffnInputSumMaximumAbsoluteError: number;
  ffnInputSumMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: 0;
  cpuReadbacksBetweenKernels: 0;
  gpuCopiesBetweenKernels: 0;
}

export interface DecodeAttentionBlockPipelines {
  profile: GemmaLayerProfile;
  headDim: 256 | 512;
  qOutFeatures: 2048 | 4096;
  kvOutFeatures: 256 | 512;
  qkvWorkgroupCount: 1280 | 2560;
  oprojWorkgroupCount: 192;
  rms: GPUComputePipeline;
  qkv: GPUComputePipeline;
  kNormRope: GPUComputePipeline;
  vNorm: GPUComputePipeline;
  attention: GPUComputePipeline;
  oprojNorm: GPUComputePipeline;
}

export interface DecodeAttentionBlockResources {
  rmsBindGroup: GPUBindGroup | null;
  qkvBindGroup: GPUBindGroup;
  kNormRopeBindGroup: GPUBindGroup | null;
  vNormBindGroup: GPUBindGroup | null;
  attentionBindGroup: GPUBindGroup;
  oprojNormBindGroup: GPUBindGroup;
  qkvOutputBuffer: GPUBuffer;
  attentionOutputBuffer: GPUBuffer;
  hiddenBuffer: GPUBuffer;
  ffnInputBuffer: GPUBuffer;
  ffnInputSumBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  cache: DecodeKvCache;
  cachePosition: number;
  cosineBuffer: GPUBuffer;
  sineBuffer: GPUBuffer;
  qkvParamsBuffer: GPUBuffer;
  kNormParamsBuffer: GPUBuffer;
  vNormParamsBuffer: GPUBuffer;
  attentionParamsBuffer: GPUBuffer;
  modelWeights: DecodeAttentionModelWeightBuffers;
  modelScales: DecodeAttentionModelScales;
  runsInputRms: boolean;
  writesKvCache: boolean;
  ownsCache: boolean;
  qkvWorkgroupCount: number;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

export interface DecodeAttentionModelWeightBuffers {
  inputNorm: GPUBuffer | null;
  qkvPacked: GPUBuffer;
  qkvRowScales: GPUBuffer;
  kNorm: GPUBuffer | null;
  qNorm: GPUBuffer;
  outputProjectionPacked: GPUBuffer;
  outputProjectionRowScales: GPUBuffer;
  postAttentionAndPreFeedforwardNorm: GPUBuffer;
}

export interface DecodeAttentionModelScales {
  qkvInput: number;
  qkvOutput: Float32Array;
  attentionOutput: number;
  outputProjectionOutput: number;
  preMlpInput: number;
}

export interface DecodeAttentionActivationBuffers {
  input: GPUBuffer;
  inputSum: GPUBuffer;
  hidden: GPUBuffer;
}

export interface DecodeAttentionRuntimeInputs {
  hidden: Float32Array;
  hiddenBuffer?: GPUBuffer;
  cosine: Float32Array;
  sine: Float32Array;
  keyCache: Float32Array;
  valueCache: Float32Array;
  keyLength: number;
  cacheCapacity?: number;
  queryOffset: number;
  qHeads: 8;
  kvHeads: 1;
  window: number;
}

export interface DecodeSharedKvAttentionRuntimeInputs extends Omit<
  DecodeAttentionRuntimeInputs,
  "keyCache" | "valueCache"
> {
  sourceCache: DecodeKvCache;
}

interface DecodeAttentionBlockWeights {
  layerIndex: number;
  rmsWeight: Float32Array;
  rmsInputScale: number;
  qkvWeights: Uint32Array;
  qkvScales: Float32Array;
  qkvOutputScales: Float32Array;
  kNormWeight: Float32Array | null;
  qNormWeight: Float32Array;
  attentionOutputScale: number;
  oprojWeights: Uint32Array;
  oprojScales: Float32Array;
  oprojNormWeights: Float32Array;
  oprojOutputScale: number;
  preMlpInputScale: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<DecodeAttentionBlockPipelines>>();

export async function benchmarkDecodeAttentionBlock(
  iterations = 20,
): Promise<DecodeAttentionBlockBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }
  const [rmsFixture, qkvFixture, kNormFixture, attentionFixture, oprojFixture, device] =
    await Promise.all([
      loadDecodeRmsFixture(),
      loadCapturedQatQkvFixture(),
      loadDecodeKNormRopeFixture(),
      loadDecodeAttentionFixture(),
      loadDecodeOprojNormFixture(),
      getWebGpuDevice(),
    ]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("Decode attention block requires WebGPU subgroups and shader-f16");
  }
  assertExact(rmsFixture.expectedOutput, qkvFixture.input, "RMS/SRQ to QKV activation");
  assertExact(rmsFixture.expectedSum, qkvFixture.inputSum, "RMS/SRQ to QKV sumA");
  assertExact(qkvFixture.expectedQ, attentionFixture.q, "Q projection to attention");
  assertExact(qkvFixture.expectedK, kNormFixture.input, "K projection to K norm");
  assertExact(kNormFixture.expectedOutput, attentionFixture.keyCache.subarray(10 * KV_OUT), "K norm to cache");
  assertExact(kNormFixture.cosine, attentionFixture.cosine, "K/Q cosine row");
  assertExact(kNormFixture.sine, attentionFixture.sine, "K/Q sine row");
  assertExact(attentionFixture.expectedOutput, oprojFixture.attention, "Attention to O projection");
  assertExact(rmsFixture.hidden, oprojFixture.hiddenBefore, "Layer residual hidden state");

  const cached = pipelineCache.get(device);
  const compiledPromise = cached ?? compileDecodeAttentionBlockPipelines(device);
  if (!cached) pipelineCache.set(device, compiledPromise);
  let pipelines: DecodeAttentionBlockPipelines;
  try {
    pipelines = await compiledPromise;
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }

  const resources = createDecodeAttentionBlockResources(
    device,
    pipelines,
    rmsFixture,
    qkvFixture,
    kNormFixture,
    attentionFixture,
    oprojFixture,
  );
  try {
    device.queue.writeBuffer(resources.hiddenBuffer, 0, rmsFixture.hidden);
    await dispatch(device, pipelines, resources);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      device.queue.writeBuffer(resources.hiddenBuffer, 0, rmsFixture.hidden);
      const started = performance.now();
      await dispatch(device, pipelines, resources);
      dispatchSamples.push(performance.now() - started);
    }
    device.queue.writeBuffer(resources.hiddenBuffer, 0, rmsFixture.hidden);
    await dispatch(device, pipelines, resources);

    const qBytes = Q_OUT * 4;
    const kvBytes = KV_OUT * 4;
    const attentionBytes = attentionFixture.expectedOutput.byteLength;
    const hiddenBytes = oprojFixture.expectedHidden.byteLength;
    const ffnInputBytes = oprojFixture.expectedFfnInputBits.byteLength;
    const ffnInputSumBytes = oprojFixture.expectedFfnInputSum.byteLength;
    const hiddenOffset = qBytes + 3 * kvBytes + attentionBytes;
    const ffnInputOffset = hiddenOffset + hiddenBytes;
    const ffnInputSumOffset = ffnInputOffset + ffnInputBytes;
    const encoder = device.createCommandEncoder({ label: "Decode attention block readback" });
    encoder.copyBufferToBuffer(resources.qkvOutputBuffer, 0, resources.readBuffer, 0, qBytes + kvBytes);
    encoder.copyBufferToBuffer(
      resources.cache.keyBuffer,
      resources.cache.byteOffset(resources.cachePosition),
      resources.readBuffer,
      qBytes + kvBytes,
      kvBytes,
    );
    encoder.copyBufferToBuffer(
      resources.cache.valueBuffer,
      resources.cache.byteOffset(resources.cachePosition),
      resources.readBuffer,
      qBytes + 2 * kvBytes,
      kvBytes,
    );
    encoder.copyBufferToBuffer(
      resources.attentionOutputBuffer,
      0,
      resources.readBuffer,
      qBytes + 3 * kvBytes,
      attentionBytes,
    );
    encoder.copyBufferToBuffer(
      resources.hiddenBuffer,
      0,
      resources.readBuffer,
      hiddenOffset,
      hiddenBytes,
    );
    encoder.copyBufferToBuffer(
      resources.ffnInputBuffer,
      0,
      resources.readBuffer,
      ffnInputOffset,
      ffnInputBytes,
    );
    encoder.copyBufferToBuffer(
      resources.ffnInputSumBuffer,
      0,
      resources.readBuffer,
      ffnInputSumOffset,
      ffnInputSumBytes,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const output = resources.readBuffer.getMappedRange();
    const actualQ = new Float32Array(output.slice(0, qBytes));
    const actualRawK = new Float32Array(output.slice(qBytes, qBytes + kvBytes));
    const actualCachedK = new Float32Array(output.slice(qBytes + kvBytes, qBytes + 2 * kvBytes));
    const actualCachedV = new Float32Array(output.slice(qBytes + 2 * kvBytes, qBytes + 3 * kvBytes));
    const actualAttention = new Float32Array(output.slice(qBytes + 3 * kvBytes, hiddenOffset));
    const actualHidden = new Float32Array(output.slice(hiddenOffset, ffnInputOffset));
    const actualFfnInputBits = new Uint16Array(output.slice(ffnInputOffset, ffnInputSumOffset));
    const actualFfnInputSum = new Float32Array(output.slice(ffnInputSumOffset));
    resources.readBuffer.unmap();

    const qErrors = measureErrors(actualQ, qkvFixture.expectedQ);
    const rawKErrors = measureErrors(actualRawK, qkvFixture.expectedK);
    const cachedKErrors = measureErrors(actualCachedK, kNormFixture.expectedOutput);
    const cachedVErrors = measureErrors(
      actualCachedV,
      attentionFixture.valueCache.subarray(resources.cachePosition * KV_OUT),
    );
    const attentionErrors = measureErrors(actualAttention, attentionFixture.expectedOutput);
    const hiddenErrors = measureErrors(actualHidden, oprojFixture.expectedHidden);
    const ffnInputSumErrors = measureErrors(
      actualFfnInputSum,
      oprojFixture.expectedFfnInputSum,
    );
    let ffnInputBitMismatches = 0;
    for (let index = 0; index < actualFfnInputBits.length; index += 1) {
      if (actualFfnInputBits[index] !== oprojFixture.expectedFfnInputBits[index]) {
        ffnInputBitMismatches += 1;
      }
    }
    const sortedSamples = dispatchSamples.toSorted((left, right) => left - right);
    return {
      sourceOperators: [
        "com.xenova.gemma4.DecodeRmsSrq",
        "com.xenova.gemma4.DecodeQkvProj",
        "com.xenova.gemma4.DecodeQkNormRope",
        "RMSNorm",
        "Gemma4DecodeAttentionPartial",
        "com.xenova.gemma4.DecodeOprojNorm",
      ],
      implementation: "shared-qkv-reusable-kv-cache-oproj",
      cacheCapacity: resources.cache.capacity,
      cacheLength: resources.cache.length,
      cachePosition: resources.cachePosition,
      dispatchesPerToken: 6,
      iterations,
      dispatchMedianMs: round(percentile(sortedSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedSamples, 0.95)),
      qMaximumAbsoluteError: qErrors.maximumAbsoluteError,
      qMaximumRelativeError: qErrors.maximumRelativeError,
      rawKMaximumAbsoluteError: rawKErrors.maximumAbsoluteError,
      rawKMaximumRelativeError: rawKErrors.maximumRelativeError,
      cachedKMaximumAbsoluteError: cachedKErrors.maximumAbsoluteError,
      cachedKMaximumRelativeError: cachedKErrors.maximumRelativeError,
      cachedVMaximumAbsoluteError: cachedVErrors.maximumAbsoluteError,
      cachedVMaximumRelativeError: cachedVErrors.maximumRelativeError,
      attentionMaximumAbsoluteError: attentionErrors.maximumAbsoluteError,
      attentionMaximumRelativeError: attentionErrors.maximumRelativeError,
      hiddenMaximumAbsoluteError: hiddenErrors.maximumAbsoluteError,
      hiddenMaximumRelativeError: hiddenErrors.maximumRelativeError,
      ffnInputBitMismatches,
      ffnInputSumMaximumAbsoluteError: ffnInputSumErrors.maximumAbsoluteError,
      ffnInputSumMaximumRelativeError: ffnInputSumErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length + resources.cache.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
      cpuReadbacksBetweenKernels: 0,
      gpuCopiesBetweenKernels: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
    resources.cache.destroy();
  }
}

export async function compileDecodeAttentionBlockPipelines(
  device: GPUDevice,
  profile: GemmaLayerProfile = "sliding-int4",
): Promise<DecodeAttentionBlockPipelines> {
  const fullAttention = profile.startsWith("full");
  const headDim = fullAttention ? 512 : 256;
  const qOutFeatures = fullAttention ? 4096 : 2048;
  const kvOutFeatures = fullAttention ? 512 : 256;
  const rmsModule = device.createShaderModule({ code: createDecodeRmsSrqShader(true) });
  const qkvModule = device.createShaderModule({
    code: createDecodeQkvPresrqCacheShader(true, qOutFeatures, kvOutFeatures),
  });
  const kNormModule = device.createShaderModule({
    code: createDecodeKNormRopeShader(headDim),
  });
  const vNormModule = device.createShaderModule({ code: createDecodeVRmsShader() });
  const attentionModule = device.createShaderModule({
    code: createDecodeAttentionShader(headDim),
  });
  const oprojNormModule = device.createShaderModule({
    code: createDecodeOprojNormShader(qOutFeatures),
  });
  const [rms, qkv, kNormRope, vNorm, attention, oprojNorm] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Decode block RMS SRQ",
      layout: "auto",
      compute: { module: rmsModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Decode block QKV direct V cache",
      layout: "auto",
      compute: { module: qkvModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Decode block K norm RoPE",
      layout: "auto",
      compute: { module: kNormModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Decode block V RMSNorm",
      layout: "auto",
      compute: { module: vNormModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Decode block attention",
      layout: "auto",
      compute: { module: attentionModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Decode block O projection norm",
      layout: "auto",
      compute: { module: oprojNormModule, entryPoint: "main" },
    }),
  ]);
  return {
    profile,
    headDim,
    qOutFeatures,
    kvOutFeatures,
    qkvWorkgroupCount: fullAttention ? 2560 : 1280,
    oprojWorkgroupCount: OPROJ_WORKGROUP_COUNT,
    rms,
    qkv,
    kNormRope,
    vNorm,
    attention,
    oprojNorm,
  };
}

export function createDecodeAttentionBlockResources(
  device: GPUDevice,
  pipelines: DecodeAttentionBlockPipelines,
  rmsFixture: Awaited<ReturnType<typeof loadDecodeRmsFixture>>,
  qkvFixture: Awaited<ReturnType<typeof loadCapturedQatQkvFixture>>,
  kNormFixture: Awaited<ReturnType<typeof loadDecodeKNormRopeFixture>>,
  attentionFixture: Awaited<ReturnType<typeof loadDecodeAttentionFixture>>,
  oprojFixture: Awaited<ReturnType<typeof loadDecodeOprojNormFixture>>,
  materialized?: MaterializedGemmaLayer,
): DecodeAttentionBlockResources {
  if (materialized && materialized.profile !== pipelines.profile) {
    throw new Error(
      `${pipelines.profile} attention pipelines do not match ${materialized.profile} weights`,
    );
  }
  return createAttentionBlockResources(device, pipelines, {
    layerIndex: materialized?.layerIndex ?? 0,
    rmsWeight: materialized?.norms.input ?? rmsFixture.weight,
    rmsInputScale: materialized?.qkv.inputScale ?? rmsFixture.inputScale,
    qkvWeights: materialized?.qkv.packedWeights ?? qkvFixture.packedWeights,
    qkvScales: materialized?.qkv.rowScales ?? qkvFixture.rowScales,
    qkvOutputScales: materialized?.qkv.outputScales ?? qkvFixture.outputScales,
    kNormWeight: materialized?.norms.k ?? kNormFixture.weight,
    qNormWeight: materialized?.norms.q ?? attentionFixture.qNormWeight,
    attentionOutputScale: materialized?.outputProjection.inputScale ??
      attentionFixture.outputQuantScale,
    oprojWeights: materialized?.outputProjection.packedWeights ?? oprojFixture.packedWeights,
    oprojScales: materialized?.outputProjection.rowScales ?? oprojFixture.rowScales,
    oprojNormWeights: materialized?.norms.oProjectionFused ?? oprojFixture.normWeights,
    oprojOutputScale: materialized?.outputProjection.outputScale ?? oprojFixture.outputScale,
    preMlpInputScale: materialized?.mlp.gate.inputScale ?? oprojFixture.inScale2,
  }, {
    hidden: rmsFixture.hidden,
    cosine: attentionFixture.cosine,
    sine: attentionFixture.sine,
    keyCache: attentionFixture.keyCache,
    valueCache: attentionFixture.valueCache,
    keyLength: attentionFixture.keyLength,
    queryOffset: attentionFixture.queryOffset,
    qHeads: attentionFixture.qHeads,
    kvHeads: attentionFixture.kvHeads,
    window: attentionFixture.window,
  });
}

export function createGemmaDecodeAttentionBlockResources(
  device: GPUDevice,
  pipelines: DecodeAttentionBlockPipelines,
  layer: MaterializedGemmaLayer,
  runtime: DecodeAttentionRuntimeInputs,
  activations?: DecodeAttentionActivationBuffers,
): DecodeAttentionBlockResources {
  if (layer.profile !== pipelines.profile) {
    throw new Error(
      `${pipelines.profile} attention pipelines do not match ${layer.profile} weights`,
    );
  }
  if (!layer.norms.k) {
    throw new Error(`Gemma layer ${layer.layerIndex} requires a shared K/V attention path`);
  }
  return createAttentionBlockResources(device, pipelines, {
    layerIndex: layer.layerIndex,
    rmsWeight: layer.norms.input,
    rmsInputScale: layer.qkv.inputScale,
    qkvWeights: layer.qkv.packedWeights,
    qkvScales: layer.qkv.rowScales,
    qkvOutputScales: layer.qkv.outputScales,
    kNormWeight: layer.norms.k,
    qNormWeight: layer.norms.q,
    attentionOutputScale: layer.outputProjection.inputScale,
    oprojWeights: layer.outputProjection.packedWeights,
    oprojScales: layer.outputProjection.rowScales,
    oprojNormWeights: layer.norms.oProjectionFused,
    oprojOutputScale: layer.outputProjection.outputScale,
    preMlpInputScale: layer.mlp.gate.inputScale,
  }, runtime, true, undefined, activations);
}

export function createGemmaDecodeSharedKvAttentionBlockResources(
  device: GPUDevice,
  pipelines: DecodeAttentionBlockPipelines,
  layer: MaterializedGemmaLayer,
  runtime: DecodeSharedKvAttentionRuntimeInputs,
  activations?: DecodeAttentionActivationBuffers,
): DecodeAttentionBlockResources {
  if (layer.profile !== pipelines.profile) {
    throw new Error(
      `${pipelines.profile} attention pipelines do not match ${layer.profile} weights`,
    );
  }
  if (layer.norms.k) {
    throw new Error(`Gemma layer ${layer.layerIndex} owns K/V and cannot use a shared cache`);
  }
  return createAttentionBlockResources(device, pipelines, {
    layerIndex: layer.layerIndex,
    rmsWeight: layer.norms.input,
    rmsInputScale: layer.qkv.inputScale,
    qkvWeights: layer.qkv.packedWeights,
    qkvScales: layer.qkv.rowScales,
    qkvOutputScales: layer.qkv.outputScales,
    kNormWeight: null,
    qNormWeight: layer.norms.q,
    attentionOutputScale: layer.outputProjection.inputScale,
    oprojWeights: layer.outputProjection.packedWeights,
    oprojScales: layer.outputProjection.rowScales,
    oprojNormWeights: layer.norms.oProjectionFused,
    oprojOutputScale: layer.outputProjection.outputScale,
    preMlpInputScale: layer.mlp.gate.inputScale,
  }, {
    hidden: runtime.hidden,
    cosine: runtime.cosine,
    sine: runtime.sine,
    keyCache: new Float32Array(0),
    valueCache: new Float32Array(0),
    keyLength: runtime.keyLength,
    cacheCapacity: runtime.cacheCapacity,
    queryOffset: runtime.queryOffset,
    qHeads: runtime.qHeads,
    kvHeads: runtime.kvHeads,
    window: runtime.window,
  }, false, runtime.sourceCache, activations);
}

function createAttentionBlockResources(
  device: GPUDevice,
  pipelines: DecodeAttentionBlockPipelines,
  weights: DecodeAttentionBlockWeights,
  runtime: DecodeAttentionRuntimeInputs,
  writesKvCache = true,
  sourceCache?: DecodeKvCache,
  activations?: DecodeAttentionActivationBuffers,
): DecodeAttentionBlockResources {
  if (runtime.qHeads * pipelines.headDim !== pipelines.qOutFeatures ||
      runtime.kvHeads * pipelines.headDim !== pipelines.kvOutFeatures) {
    throw new Error(`Runtime attention geometry does not match ${pipelines.profile}`);
  }
  const cachePosition = runtime.queryOffset;
  if (writesKvCache && !weights.kNormWeight) {
    throw new Error(`Gemma layer ${weights.layerIndex} is missing its K norm`);
  }
  const ownsCache = !sourceCache;
  const cacheAllocation = resolveDecodeKvCacheAllocation(runtime);
  const cache = sourceCache ?? new DecodeKvCache(device, {
      capacity: cacheAllocation.capacity,
      kvHeads: runtime.kvHeads,
      headDim: pipelines.headDim,
      mode: cacheAllocation.mode,
      label: `Layer ${weights.layerIndex} decode block cache`,
    });
  if (cache.kvHeads !== runtime.kvHeads || cache.headDim !== pipelines.headDim ||
      (cache.mode === "linear" && cache.capacity < runtime.keyLength)) {
    throw new Error(`Shared K/V cache geometry does not match ${pipelines.profile}`);
  }
  const prefixElements = cachePosition * cache.tokenElements;
  if (ownsCache && prefixElements > 0) {
    cache.writeTokens(
      device.queue,
      0,
      runtime.keyCache.subarray(0, prefixElements),
      runtime.valueCache.subarray(0, prefixElements),
    );
  }

  const runsInputRms = !activations;
  const ownsHiddenBuffer = !activations && !runtime.hiddenBuffer;
  const hiddenBuffer = activations?.hidden ?? runtime.hiddenBuffer ??
    storageBuffer(device, "Decode block hidden", runtime.hidden.byteLength, true, true);
  const rmsWeightBuffer = runsInputRms
    ? storageBuffer(device, "Decode block RMS weight", weights.rmsWeight.byteLength, true)
    : null;
  const rmsOutputBuffer = runsInputRms
    ? storageBuffer(device, "Decode block RMS output", runtime.hidden.byteLength, false)
    : activations.input;
  const rmsSumBuffer = runsInputRms
    ? storageBuffer(device, "Decode block RMS sum", 4, false)
    : activations.inputSum;
  const rmsParamsBuffer = runsInputRms
    ? uniformBuffer(device, "Decode block RMS parameters", 16)
    : null;
  const weightBuffer = storageBuffer(device, "Decode block QKV weights", weights.qkvWeights.byteLength, true);
  const scaleBuffer = storageBuffer(device, "Decode block QKV scales", weights.qkvScales.byteLength, true);
  const qkvOutputElements = pipelines.qOutFeatures +
    (writesKvCache ? pipelines.kvOutFeatures : 0);
  const qkvOutputBuffer = storageBuffer(device, "Decode block Q and raw K", qkvOutputElements * 4, false, true);
  const qkvParamsBuffer = uniformBuffer(device, "Decode block QKV parameters", 16);
  const kNormWeightBuffer = weights.kNormWeight
    ? storageBuffer(device, "Decode block K norm weight", weights.kNormWeight.byteLength, true)
    : null;
  const cosineBuffer = storageBuffer(device, "Decode block cosine", runtime.cosine.byteLength, true);
  const sineBuffer = storageBuffer(device, "Decode block sine", runtime.sine.byteLength, true);
  const kNormParamsBuffer = uniformBuffer(device, "Decode block K norm parameters", 16);
  const vNormParamsBuffer = uniformBuffer(device, "Decode block V norm parameters", 16);
  const qNormWeightBuffer = storageBuffer(device, "Decode block Q norm weight", weights.qNormWeight.byteLength, true);
  const partialElements = runtime.qHeads * ATTENTION_CHUNK_COUNT *
    (pipelines.headDim + 2) + runtime.qHeads;
  const partialBuffer = storageBuffer(device, "Decode block attention partials", partialElements * 4, false);
  const attentionOutputBuffer = storageBuffer(device, "Decode block attention output", pipelines.qOutFeatures * 4, false, true);
  const attentionParamsBuffer = uniformBuffer(device, "Decode block attention parameters", 32);
  const oprojWeightBuffer = storageBuffer(device, "Decode block O projection weights", weights.oprojWeights.byteLength, true);
  const oprojScaleBuffer = storageBuffer(device, "Decode block O projection scales", weights.oprojScales.byteLength, true);
  const oprojNormWeightBuffer = storageBuffer(device, "Decode block O projection norm weights", weights.oprojNormWeights.byteLength, true);
  const oprojPartialBuffer = storageBuffer(device, "Decode block O projection partials", 1537 * 4, false);
  const ffnInputBuffer = storageBuffer(device, "Decode block FFN input", runtime.hidden.length * Uint16Array.BYTES_PER_ELEMENT, false, true);
  const ffnInputSumBuffer = storageBuffer(device, "Decode block FFN input sum", 4, false, true);
  const oprojParamsBuffer = uniformBuffer(device, "Decode block O projection parameters", 16);
  const readBuffer = device.createBuffer({
    label: "Decode block readback",
    size:
      (pipelines.qOutFeatures * 2 + pipelines.kvOutFeatures * 3) * 4 +
      runtime.hidden.byteLength +
      runtime.hidden.length * Uint16Array.BYTES_PER_ELEMENT + 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const buffers = [
    ...(activations ? [] : [
      ...(ownsHiddenBuffer ? [hiddenBuffer] : []),
      rmsWeightBuffer!,
      rmsOutputBuffer,
      rmsSumBuffer,
      rmsParamsBuffer!,
    ]),
    weightBuffer,
    scaleBuffer,
    qkvOutputBuffer,
    qkvParamsBuffer,
    ...(kNormWeightBuffer ? [kNormWeightBuffer] : []),
    cosineBuffer,
    sineBuffer,
    kNormParamsBuffer,
    vNormParamsBuffer,
    qNormWeightBuffer,
    partialBuffer,
    attentionOutputBuffer,
    attentionParamsBuffer,
    oprojWeightBuffer,
    oprojScaleBuffer,
    oprojNormWeightBuffer,
    oprojPartialBuffer,
    ffnInputBuffer,
    ffnInputSumBuffer,
    oprojParamsBuffer,
    readBuffer,
  ];

  const rmsParams = new ArrayBuffer(16);
  const rmsParamsView = new DataView(rmsParams);
  rmsParamsView.setUint32(0, 1, true);
  rmsParamsView.setUint32(4, 1, true);
  rmsParamsView.setFloat32(8, weights.rmsInputScale, true);
  if (runsInputRms && rmsWeightBuffer && rmsParamsBuffer) {
    if (ownsHiddenBuffer) device.queue.writeBuffer(hiddenBuffer, 0, runtime.hidden);
    device.queue.writeBuffer(rmsWeightBuffer, 0, weights.rmsWeight);
    device.queue.writeBuffer(rmsParamsBuffer, 0, rmsParams);
  }
  device.queue.writeBuffer(weightBuffer, 0, weights.qkvWeights);
  device.queue.writeBuffer(scaleBuffer, 0, weights.qkvScales);
  device.queue.writeBuffer(qkvParamsBuffer, 0, weights.qkvOutputScales);
  device.queue.writeBuffer(qkvParamsBuffer, 12, new Uint32Array([cache.elementOffset(cachePosition)]));
  if (kNormWeightBuffer && weights.kNormWeight) {
    device.queue.writeBuffer(kNormWeightBuffer, 0, weights.kNormWeight);
  }
  device.queue.writeBuffer(cosineBuffer, 0, runtime.cosine);
  device.queue.writeBuffer(sineBuffer, 0, runtime.sine);
  device.queue.writeBuffer(
    kNormParamsBuffer,
    0,
    new Uint32Array([1, 1, cache.elementOffset(cachePosition), 0]),
  );
  device.queue.writeBuffer(
    vNormParamsBuffer,
    0,
    new Uint32Array([1, pipelines.kvOutFeatures, cache.elementOffset(cachePosition), 0]),
  );
  device.queue.writeBuffer(qNormWeightBuffer, 0, weights.qNormWeight);
  const attentionParams = new ArrayBuffer(32);
  const attentionParamsView = new DataView(attentionParams);
  attentionParamsView.setUint32(0, 1, true);
  attentionParamsView.setUint32(4, runtime.keyLength, true);
  attentionParamsView.setUint32(8, runtime.queryOffset, true);
  attentionParamsView.setUint32(12, runtime.qHeads, true);
  attentionParamsView.setUint32(16, runtime.kvHeads, true);
  attentionParamsView.setUint32(20, runtime.window, true);
  attentionParamsView.setFloat32(24, weights.attentionOutputScale, true);
  attentionParamsView.setUint32(28, cache.capacity, true);
  device.queue.writeBuffer(attentionParamsBuffer, 0, attentionParams);
  device.queue.writeBuffer(oprojWeightBuffer, 0, weights.oprojWeights);
  device.queue.writeBuffer(oprojScaleBuffer, 0, weights.oprojScales);
  device.queue.writeBuffer(oprojNormWeightBuffer, 0, weights.oprojNormWeights);
  device.queue.writeBuffer(
    oprojParamsBuffer,
    0,
    new Float32Array([weights.oprojOutputScale, weights.preMlpInputScale]),
  );

  return {
    rmsBindGroup: runsInputRms && rmsWeightBuffer && rmsParamsBuffer
      ? device.createBindGroup({
      layout: pipelines.rms.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: rmsWeightBuffer } },
        { binding: 2, resource: { buffer: rmsOutputBuffer } },
        { binding: 3, resource: { buffer: rmsSumBuffer } },
        { binding: 4, resource: { buffer: rmsParamsBuffer } },
      ],
      })
      : null,
    qkvBindGroup: device.createBindGroup({
      layout: pipelines.qkv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rmsOutputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: rmsSumBuffer } },
        { binding: 4, resource: { buffer: qkvOutputBuffer } },
        { binding: 5, resource: { buffer: qkvParamsBuffer } },
        { binding: 6, resource: { buffer: cache.valueBuffer } },
      ],
    }),
    kNormRopeBindGroup: kNormWeightBuffer ? device.createBindGroup({
      layout: pipelines.kNormRope.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qkvOutputBuffer, offset: pipelines.qOutFeatures * 4, size: pipelines.kvOutFeatures * 4 } },
        { binding: 1, resource: { buffer: kNormWeightBuffer } },
        { binding: 2, resource: { buffer: cosineBuffer } },
        { binding: 3, resource: { buffer: sineBuffer } },
        { binding: 4, resource: { buffer: cache.keyBuffer } },
        { binding: 5, resource: { buffer: kNormParamsBuffer } },
      ],
    }) : null,
    vNormBindGroup: writesKvCache ? device.createBindGroup({
      layout: pipelines.vNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cache.valueBuffer } },
        { binding: 1, resource: { buffer: vNormParamsBuffer } },
      ],
    }) : null,
    attentionBindGroup: device.createBindGroup({
      layout: pipelines.attention.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qkvOutputBuffer, size: pipelines.qOutFeatures * 4 } },
        { binding: 1, resource: { buffer: qNormWeightBuffer } },
        { binding: 2, resource: { buffer: cosineBuffer } },
        { binding: 3, resource: { buffer: sineBuffer } },
        { binding: 4, resource: { buffer: cache.keyBuffer } },
        { binding: 5, resource: { buffer: cache.valueBuffer } },
        { binding: 6, resource: { buffer: partialBuffer } },
        { binding: 7, resource: { buffer: attentionOutputBuffer } },
        { binding: 8, resource: { buffer: attentionParamsBuffer } },
      ],
    }),
    oprojNormBindGroup: device.createBindGroup({
      layout: pipelines.oprojNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: attentionOutputBuffer } },
        { binding: 1, resource: { buffer: oprojWeightBuffer } },
        { binding: 2, resource: { buffer: oprojScaleBuffer } },
        { binding: 3, resource: { buffer: hiddenBuffer } },
        { binding: 4, resource: { buffer: oprojNormWeightBuffer } },
        { binding: 5, resource: { buffer: oprojPartialBuffer } },
        { binding: 6, resource: { buffer: ffnInputBuffer } },
        { binding: 7, resource: { buffer: ffnInputSumBuffer } },
        { binding: 8, resource: { buffer: oprojParamsBuffer } },
      ],
    }),
    qkvOutputBuffer,
    attentionOutputBuffer,
    hiddenBuffer,
    ffnInputBuffer,
    ffnInputSumBuffer,
    readBuffer,
    cache,
    cachePosition,
    cosineBuffer,
    sineBuffer,
    qkvParamsBuffer,
    kNormParamsBuffer,
    vNormParamsBuffer,
    attentionParamsBuffer,
    modelWeights: {
      inputNorm: rmsWeightBuffer,
      qkvPacked: weightBuffer,
      qkvRowScales: scaleBuffer,
      kNorm: kNormWeightBuffer,
      qNorm: qNormWeightBuffer,
      outputProjectionPacked: oprojWeightBuffer,
      outputProjectionRowScales: oprojScaleBuffer,
      postAttentionAndPreFeedforwardNorm: oprojNormWeightBuffer,
    },
    modelScales: {
      qkvInput: weights.rmsInputScale,
      qkvOutput: weights.qkvOutputScales,
      attentionOutput: weights.attentionOutputScale,
      outputProjectionOutput: weights.oprojOutputScale,
      preMlpInput: weights.preMlpInputScale,
    },
    runsInputRms,
    writesKvCache,
    ownsCache,
    qkvWorkgroupCount: writesKvCache
      ? pipelines.qkvWorkgroupCount
      : pipelines.qOutFeatures / 2,
    buffers,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0) +
      (ownsCache ? cache.bytesAllocated : 0),
  };
}

export function updateDecodeAttentionBlockToken(
  device: GPUDevice,
  resources: DecodeAttentionBlockResources,
  position: number,
  cosine: Float32Array,
  sine: Float32Array,
): void {
  if (!Number.isInteger(position) || position < 0 ||
      (resources.cache.mode === "linear" && position >= resources.cache.capacity)) {
    throw new Error("Decode attention position exceeds cache capacity");
  }
  if (resources.cache.length !== position) {
    throw new Error(
      `Decode attention position ${position} does not follow cache length ${resources.cache.length}`,
    );
  }
  if (cosine.byteLength !== resources.cosineBuffer.size ||
      sine.byteLength !== resources.sineBuffer.size) {
    throw new Error("Decode attention rotary row does not match its profile");
  }
  const destinationOffset = resources.cache.elementOffset(position);
  device.queue.writeBuffer(resources.cosineBuffer, 0, cosine);
  device.queue.writeBuffer(resources.sineBuffer, 0, sine);
  device.queue.writeBuffer(resources.qkvParamsBuffer, 12, new Uint32Array([destinationOffset]));
  device.queue.writeBuffer(resources.kNormParamsBuffer, 8, new Uint32Array([destinationOffset]));
  device.queue.writeBuffer(resources.vNormParamsBuffer, 8, new Uint32Array([destinationOffset]));
  device.queue.writeBuffer(
    resources.attentionParamsBuffer,
    4,
    new Uint32Array([position + 1, position]),
  );
  resources.cachePosition = position;
}

export function commitDecodeAttentionBlockCache(
  resources: DecodeAttentionBlockResources,
): void {
  if (resources.writesKvCache) {
    resources.cache.commitWrite(resources.cachePosition);
  }
}

export function destroyDecodeAttentionBlockResources(
  resources: DecodeAttentionBlockResources,
): void {
  for (const buffer of resources.buffers) buffer.destroy();
  if (resources.ownsCache) resources.cache.destroy();
}

async function dispatch(
  device: GPUDevice,
  pipelines: DecodeAttentionBlockPipelines,
  resources: DecodeAttentionBlockResources,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "Decode attention block" });
  encodeDecodeAttentionBlock(encoder, pipelines, resources);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  commitDecodeAttentionBlockCache(resources);
}

export function encodeDecodeAttentionBlock(
  encoder: GPUCommandEncoder,
  pipelines: DecodeAttentionBlockPipelines,
  resources: DecodeAttentionBlockResources,
): void {
  if (resources.runsInputRms) {
    if (!resources.rmsBindGroup) {
      throw new Error("Initial attention resources are missing their RMS bind group");
    }
    const rmsPass = encoder.beginComputePass({ label: "Decode block RMS SRQ" });
    rmsPass.setPipeline(pipelines.rms);
    rmsPass.setBindGroup(0, resources.rmsBindGroup);
    rmsPass.dispatchWorkgroups(1, 1, 1);
    rmsPass.end();
  }
  const qkvPass = encoder.beginComputePass({ label: "Decode block QKV" });
  qkvPass.setPipeline(pipelines.qkv);
  qkvPass.setBindGroup(0, resources.qkvBindGroup);
  qkvPass.dispatchWorkgroups(resources.qkvWorkgroupCount, 1, 1);
  qkvPass.end();
  if (resources.writesKvCache) {
    if (!resources.kNormRopeBindGroup || !resources.vNormBindGroup) {
      throw new Error("K/V-owning attention resources are incomplete");
    }
    const kNormPass = encoder.beginComputePass({ label: "Decode block K norm RoPE" });
    kNormPass.setPipeline(pipelines.kNormRope);
    kNormPass.setBindGroup(0, resources.kNormRopeBindGroup);
    kNormPass.dispatchWorkgroups(1, 1, 1);
    kNormPass.end();
    const vNormPass = encoder.beginComputePass({ label: "Decode block V RMSNorm" });
    vNormPass.setPipeline(pipelines.vNorm);
    vNormPass.setBindGroup(0, resources.vNormBindGroup);
    vNormPass.dispatchWorkgroups(1, 1, 1);
    vNormPass.end();
  }
  const attentionPass = encoder.beginComputePass({ label: "Decode block attention" });
  attentionPass.setPipeline(pipelines.attention);
  attentionPass.setBindGroup(0, resources.attentionBindGroup);
  attentionPass.dispatchWorkgroups(8, ATTENTION_CHUNK_COUNT, 1);
  attentionPass.end();
  const oprojNormPass = encoder.beginComputePass({ label: "Decode block O projection norm" });
  oprojNormPass.setPipeline(pipelines.oprojNorm);
  oprojNormPass.setBindGroup(0, resources.oprojNormBindGroup);
  oprojNormPass.dispatchWorkgroups(pipelines.oprojWorkgroupCount, 1, 1);
  oprojNormPass.end();
}

function storageBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  copyDestination: boolean,
  copySource = false,
): GPUBuffer {
  let usage = GPUBufferUsage.STORAGE;
  if (copyDestination) usage |= GPUBufferUsage.COPY_DST;
  if (copySource) usage |= GPUBufferUsage.COPY_SRC;
  return device.createBuffer({ label, size, usage });
}

function uniformBuffer(device: GPUDevice, label: string, size: number): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function assertExact(actual: Float32Array, expected: Float32Array, boundary: string): void {
  if (actual.length !== expected.length) {
    throw new Error(`${boundary} length mismatch: ${actual.length} != ${expected.length}`);
  }
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < actualBits.length; index += 1) {
    if (actualBits[index] !== expectedBits[index]) {
      throw new Error(`${boundary} differs at element ${index}`);
    }
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
  const index = Math.min(sortedValues.length - 1, Math.ceil(sortedValues.length * quantile) - 1);
  return sortedValues[index];
}

function round(value: number): number {
  return Number(value.toFixed(6));
}