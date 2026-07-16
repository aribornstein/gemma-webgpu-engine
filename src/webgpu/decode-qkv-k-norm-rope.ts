import { loadDecodeKNormRopeFixture } from "../model/decode-k-norm-rope-fixture";
import { loadCapturedQatQkvFixture } from "../model/qat-linear-fixture";
import { createDecodeKNormRopeShader } from "./decode-k-norm-rope";
import { getWebGpuDevice } from "./device";
import { createDecodeQkvPresrqShader } from "./qat-qkv-presrq";

const Q_OUT = 2048;
const KV_OUT = 256;
const QKV_WORKGROUP_COUNT = 1280;
const K_CACHE_POSITION = 10;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface DecodeQkvKNormRopeBenchmarkResult {
  sourceOperators: [
    "com.xenova.gemma4.DecodeQkvProj",
    "com.xenova.gemma4.DecodeQkNormRope",
  ];
  implementation: "shared-storage-k-cache";
  qkvArtifactSha256: string;
  kNormRopeArtifactSha256: string;
  qkvReferenceSha256: string;
  kNormRopeSourceCaptureSha256: string;
  cachePosition: number;
  cacheElementOffset: number;
  iterations: number;
  subgroupReduction: boolean;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  gpuKernelPairsPerSample: number | null;
  gpuKernelPairSamplesMs: number[] | null;
  gpuKernelPairMedianMs: number | null;
  gpuKernelPairP95Ms: number | null;
  qMaximumAbsoluteError: number;
  qMaximumRelativeError: number;
  kMaximumAbsoluteError: number;
  kMaximumRelativeError: number;
  vMaximumAbsoluteError: number;
  vMaximumRelativeError: number;
  normalizedKMaximumAbsoluteError: number;
  normalizedKMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
  cpuReadbacksBetweenKernels: 0;
  gpuCopiesBetweenKernels: 0;
}

interface CompiledPipelines {
  qkv: GPUComputePipeline;
  kNormRope: GPUComputePipeline;
  compileMs: number;
}

interface ComposedResources {
  qkvBindGroup: GPUBindGroup;
  kNormRopeBindGroup: GPUBindGroup;
  qkvOutputBuffer: GPUBuffer;
  kCacheBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipelines>>();

export async function benchmarkDecodeQkvKNormRope(
  iterations = 20,
): Promise<DecodeQkvKNormRopeBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const [qkvFixture, kNormRopeFixture, device] = await Promise.all([
    loadCapturedQatQkvFixture(),
    loadDecodeKNormRopeFixture(),
    getWebGpuDevice(),
  ]);
  assertExactBoundary(qkvFixture.expectedK, kNormRopeFixture.input);

  const subgroupReduction = device.features.has("subgroups");
  const cached = pipelineCache.get(device);
  const compiledPromise = cached ?? compilePipelines(device, subgroupReduction);
  if (!cached) pipelineCache.set(device, compiledPromise);

  let compiled: CompiledPipelines;
  try {
    compiled = await compiledPromise;
  } catch (error) {
    pipelineCache.delete(device);
    throw error;
  }

  const resources = createResources(device, compiled, qkvFixture, kNormRopeFixture);
  try {
    await dispatch(device, compiled, resources);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, compiled, resources);
      dispatchSamples.push(performance.now() - started);
    }

    const qkvBytes = (Q_OUT + 2 * KV_OUT) * 4;
    const normalizedKBytes = KV_OUT * 4;
    const cacheByteOffset = K_CACHE_POSITION * normalizedKBytes;
    const encoder = device.createCommandEncoder({ label: "Decode QKV K norm readback" });
    encoder.copyBufferToBuffer(resources.qkvOutputBuffer, 0, resources.readBuffer, 0, qkvBytes);
    encoder.copyBufferToBuffer(
      resources.kCacheBuffer,
      cacheByteOffset,
      resources.readBuffer,
      qkvBytes,
      normalizedKBytes,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const output = resources.readBuffer.getMappedRange();
    const qBytes = Q_OUT * 4;
    const kvBytes = KV_OUT * 4;
    const actualQ = new Float32Array(output.slice(0, qBytes));
    const actualK = new Float32Array(output.slice(qBytes, qBytes + kvBytes));
    const actualV = new Float32Array(output.slice(qBytes + kvBytes, qkvBytes));
    const actualNormalizedK = new Float32Array(output.slice(qkvBytes));
    resources.readBuffer.unmap();

    const qErrors = measureErrors(actualQ, qkvFixture.expectedQ);
    const kErrors = measureErrors(actualK, qkvFixture.expectedK);
    const vErrors = measureErrors(actualV, qkvFixture.expectedV);
    const normalizedKErrors = measureErrors(
      actualNormalizedK,
      kNormRopeFixture.expectedOutput,
    );
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernelPairs(device, compiled, resources);

    return {
      sourceOperators: [
        "com.xenova.gemma4.DecodeQkvProj",
        "com.xenova.gemma4.DecodeQkNormRope",
      ],
      implementation: "shared-storage-k-cache",
      qkvArtifactSha256: qkvFixture.artifactSha256,
      kNormRopeArtifactSha256: kNormRopeFixture.artifactSha256,
      qkvReferenceSha256: qkvFixture.referenceSha256,
      kNormRopeSourceCaptureSha256: kNormRopeFixture.sourceCaptureSha256,
      cachePosition: K_CACHE_POSITION,
      cacheElementOffset: K_CACHE_POSITION * KV_OUT,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelPairsPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelPairSamplesMs: gpuSamples?.map(round) ?? null,
      gpuKernelPairMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelPairP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      qMaximumAbsoluteError: qErrors.maximumAbsoluteError,
      qMaximumRelativeError: qErrors.maximumRelativeError,
      kMaximumAbsoluteError: kErrors.maximumAbsoluteError,
      kMaximumRelativeError: kErrors.maximumRelativeError,
      vMaximumAbsoluteError: vErrors.maximumAbsoluteError,
      vMaximumRelativeError: vErrors.maximumRelativeError,
      normalizedKMaximumAbsoluteError: normalizedKErrors.maximumAbsoluteError,
      normalizedKMaximumRelativeError: normalizedKErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
      cpuReadbacksBetweenKernels: 0,
      gpuCopiesBetweenKernels: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipelines(
  device: GPUDevice,
  subgroupReduction: boolean,
): Promise<CompiledPipelines> {
  const started = performance.now();
  const qkvModule = device.createShaderModule({
    code: createDecodeQkvPresrqShader(subgroupReduction),
  });
  const kNormRopeModule = device.createShaderModule({ code: createDecodeKNormRopeShader() });
  const [qkv, kNormRope] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Composed DecodeQkvProj",
      layout: "auto",
      compute: { module: qkvModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Composed DecodeQkNormRope",
      layout: "auto",
      compute: { module: kNormRopeModule, entryPoint: "main" },
    }),
  ]);
  return { qkv, kNormRope, compileMs: performance.now() - started };
}

function createResources(
  device: GPUDevice,
  pipelines: CompiledPipelines,
  qkvFixture: Awaited<ReturnType<typeof loadCapturedQatQkvFixture>>,
  kNormRopeFixture: Awaited<ReturnType<typeof loadDecodeKNormRopeFixture>>,
): ComposedResources {
  const qkvInputBuffer = createBuffer(
    device,
    "Decode QKV activation",
    qkvFixture.input.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const qkvWeightBuffer = createBuffer(
    device,
    "Decode QKV weights",
    qkvFixture.packedWeights.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const qkvScaleBuffer = createBuffer(
    device,
    "Decode QKV scales",
    qkvFixture.rowScales.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const qkvSumBuffer = createBuffer(
    device,
    "Decode QKV activation sum",
    qkvFixture.inputSum.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const qkvOutputBuffer = createBuffer(
    device,
    "Decode QKV shared output",
    (Q_OUT + 2 * KV_OUT) * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const qkvParamsBuffer = createBuffer(
    device,
    "Decode QKV parameters",
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const kNormWeightBuffer = createBuffer(
    device,
    "Decode K norm weight",
    kNormRopeFixture.weight.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const cosineBuffer = createBuffer(
    device,
    "Decode K RoPE cosine",
    kNormRopeFixture.cosine.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const sineBuffer = createBuffer(
    device,
    "Decode K RoPE sine",
    kNormRopeFixture.sine.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const kCacheBuffer = createBuffer(
    device,
    "Decode K cache",
    (K_CACHE_POSITION + 1) * KV_OUT * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const kNormRopeParamsBuffer = createBuffer(
    device,
    "Decode K norm RoPE parameters",
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const readBuffer = createBuffer(
    device,
    "Decode QKV K norm readback",
    (Q_OUT + 3 * KV_OUT) * 4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  );
  const buffers = [
    qkvInputBuffer,
    qkvWeightBuffer,
    qkvScaleBuffer,
    qkvSumBuffer,
    qkvOutputBuffer,
    qkvParamsBuffer,
    kNormWeightBuffer,
    cosineBuffer,
    sineBuffer,
    kCacheBuffer,
    kNormRopeParamsBuffer,
    readBuffer,
  ];

  device.queue.writeBuffer(qkvInputBuffer, 0, qkvFixture.input);
  device.queue.writeBuffer(qkvWeightBuffer, 0, qkvFixture.packedWeights);
  device.queue.writeBuffer(qkvScaleBuffer, 0, qkvFixture.rowScales);
  device.queue.writeBuffer(qkvSumBuffer, 0, qkvFixture.inputSum);
  device.queue.writeBuffer(qkvParamsBuffer, 0, qkvFixture.outputScales);
  device.queue.writeBuffer(kNormWeightBuffer, 0, kNormRopeFixture.weight);
  device.queue.writeBuffer(cosineBuffer, 0, kNormRopeFixture.cosine);
  device.queue.writeBuffer(sineBuffer, 0, kNormRopeFixture.sine);
  device.queue.writeBuffer(
    kNormRopeParamsBuffer,
    0,
    new Uint32Array([1, 1, K_CACHE_POSITION * KV_OUT, 0]),
  );

  return {
    qkvBindGroup: device.createBindGroup({
      layout: pipelines.qkv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qkvInputBuffer } },
        { binding: 1, resource: { buffer: qkvWeightBuffer } },
        { binding: 2, resource: { buffer: qkvScaleBuffer } },
        { binding: 3, resource: { buffer: qkvSumBuffer } },
        { binding: 4, resource: { buffer: qkvOutputBuffer } },
        { binding: 5, resource: { buffer: qkvParamsBuffer } },
      ],
    }),
    kNormRopeBindGroup: device.createBindGroup({
      layout: pipelines.kNormRope.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: qkvOutputBuffer, offset: Q_OUT * 4, size: KV_OUT * 4 },
        },
        { binding: 1, resource: { buffer: kNormWeightBuffer } },
        { binding: 2, resource: { buffer: cosineBuffer } },
        { binding: 3, resource: { buffer: sineBuffer } },
        { binding: 4, resource: { buffer: kCacheBuffer } },
        { binding: 5, resource: { buffer: kNormRopeParamsBuffer } },
      ],
    }),
    qkvOutputBuffer,
    kCacheBuffer,
    readBuffer,
    buffers,
    bytesAllocated: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
  };
}

function encodePair(
  encoder: GPUCommandEncoder,
  pipelines: CompiledPipelines,
  resources: ComposedResources,
  timestampWrites?: {
    querySet: GPUQuerySet;
    qkvBeginning?: number;
    kNormRopeEnd?: number;
  },
): void {
  const qkvPass = encoder.beginComputePass({
    label: "Composed DecodeQkvProj",
    timestampWrites: timestampWrites?.qkvBeginning === undefined ? undefined : {
      querySet: timestampWrites.querySet,
      beginningOfPassWriteIndex: timestampWrites.qkvBeginning,
    },
  });
  qkvPass.setPipeline(pipelines.qkv);
  qkvPass.setBindGroup(0, resources.qkvBindGroup);
  qkvPass.dispatchWorkgroups(QKV_WORKGROUP_COUNT);
  qkvPass.end();

  const kNormRopePass = encoder.beginComputePass({
    label: "Composed DecodeQkNormRope",
    timestampWrites: timestampWrites?.kNormRopeEnd === undefined ? undefined : {
      querySet: timestampWrites.querySet,
      endOfPassWriteIndex: timestampWrites.kNormRopeEnd,
    },
  });
  kNormRopePass.setPipeline(pipelines.kNormRope);
  kNormRopePass.setBindGroup(0, resources.kNormRopeBindGroup);
  kNormRopePass.dispatchWorkgroups(1, 1, 1);
  kNormRopePass.end();
}

async function dispatch(
  device: GPUDevice,
  pipelines: CompiledPipelines,
  resources: ComposedResources,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "Composed QKV K norm dispatch" });
  encodePair(encoder, pipelines, resources);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function measureGpuKernelPairs(
  device: GPUDevice,
  pipelines: CompiledPipelines,
  resources: ComposedResources,
): Promise<number[] | null> {
  if (!device.features.has("timestamp-query")) return null;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: "Composed QKV K norm timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "Composed QKV K norm timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < 10; sample += 1) {
      const encoder = device.createCommandEncoder({
        label: "Composed QKV K norm timestamp sample",
      });
      for (let pair = 0; pair < DISPATCHES_PER_TIMESTAMP_SAMPLE; pair += 1) {
        encodePair(encoder, pipelines, resources, {
          querySet,
          qkvBeginning: pair === 0 ? 0 : undefined,
          kNormRopeEnd: pair === DISPATCHES_PER_TIMESTAMP_SAMPLE - 1 ? 1 : undefined,
        });
      }
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

function assertExactBoundary(actual: Float32Array, expected: Float32Array): void {
  if (actual.length !== expected.length) {
    throw new Error("Decode QKV/K norm boundary length mismatch");
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(`Decode QKV/K norm boundary mismatch at ${index}`);
    }
  }
}

function createBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({ label, size, usage });
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
