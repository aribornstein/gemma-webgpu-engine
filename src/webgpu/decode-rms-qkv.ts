import { loadDecodeRmsFixture } from "../model/decode-rms-fixture";
import { loadCapturedQatQkvFixture } from "../model/qat-linear-fixture";
import { createDecodeRmsSrqShader } from "./decode-rms-srq";
import { getWebGpuDevice } from "./device";
import { createDecodeQkvPresrqShader } from "./qat-qkv-presrq";

const Q_OUT = 2048;
const KV_OUT = 256;
const QKV_WORKGROUP_COUNT = 1280;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface DecodeRmsQkvBenchmarkResult {
  sourceOperators: [
    "com.xenova.gemma4.DecodeRmsSrq",
    "com.xenova.gemma4.DecodeQkvProj",
  ];
  implementation: "shared-storage";
  rmsArtifactSha256: string;
  qkvArtifactSha256: string;
  referenceArtifactSha256: string;
  iterations: number;
  subgroupReduction: boolean;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  gpuKernelPairsPerSample: number | null;
  gpuKernelPairMedianMs: number | null;
  gpuKernelPairP95Ms: number | null;
  qMaximumAbsoluteError: number;
  qMaximumRelativeError: number;
  kMaximumAbsoluteError: number;
  kMaximumRelativeError: number;
  vMaximumAbsoluteError: number;
  vMaximumRelativeError: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: number;
  cpuReadbacksBetweenKernels: 0;
}

interface CompiledPipelines {
  rms: GPUComputePipeline;
  qkv: GPUComputePipeline;
  compileMs: number;
}

interface ComposedResources {
  rmsBindGroup: GPUBindGroup;
  qkvBindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipelines>>();

export async function benchmarkDecodeRmsQkv(
  iterations = 20,
): Promise<DecodeRmsQkvBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const [rmsFixture, qkvFixture, device] = await Promise.all([
    loadDecodeRmsFixture(),
    loadCapturedQatQkvFixture(),
    getWebGpuDevice(),
  ]);
  assertExactBoundary(rmsFixture.expectedOutput, qkvFixture.input, "activation");
  assertExactBoundary(rmsFixture.expectedSum, qkvFixture.inputSum, "sumA");

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

  const resources = createResources(device, compiled, rmsFixture, qkvFixture);
  try {
    await dispatch(device, compiled, resources);
    const dispatchSamples: number[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const started = performance.now();
      await dispatch(device, compiled, resources);
      dispatchSamples.push(performance.now() - started);
    }

    const encoder = device.createCommandEncoder({ label: "Decode RMS QKV readback" });
    encoder.copyBufferToBuffer(
      resources.outputBuffer,
      0,
      resources.readBuffer,
      0,
      resources.outputBuffer.size,
    );
    device.queue.submit([encoder.finish()]);
    await resources.readBuffer.mapAsync(GPUMapMode.READ);
    const output = resources.readBuffer.getMappedRange();
    const qBytes = Q_OUT * 4;
    const kvBytes = KV_OUT * 4;
    const actualQ = new Float32Array(output.slice(0, qBytes));
    const actualK = new Float32Array(output.slice(qBytes, qBytes + kvBytes));
    const actualV = new Float32Array(output.slice(qBytes + kvBytes));
    resources.readBuffer.unmap();

    const qErrors = measureErrors(actualQ, qkvFixture.expectedQ);
    const kErrors = measureErrors(actualK, qkvFixture.expectedK);
    const vErrors = measureErrors(actualV, qkvFixture.expectedV);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernelPairs(device, compiled, resources);

    return {
      sourceOperators: [
        "com.xenova.gemma4.DecodeRmsSrq",
        "com.xenova.gemma4.DecodeQkvProj",
      ],
      implementation: "shared-storage",
      rmsArtifactSha256: rmsFixture.artifactSha256,
      qkvArtifactSha256: qkvFixture.artifactSha256,
      referenceArtifactSha256: qkvFixture.referenceSha256,
      iterations,
      subgroupReduction,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelPairsPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelPairMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelPairP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      qMaximumAbsoluteError: qErrors.maximumAbsoluteError,
      qMaximumRelativeError: qErrors.maximumRelativeError,
      kMaximumAbsoluteError: kErrors.maximumAbsoluteError,
      kMaximumRelativeError: kErrors.maximumRelativeError,
      vMaximumAbsoluteError: vErrors.maximumAbsoluteError,
      vMaximumRelativeError: vErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
      cpuReadbacksBetweenKernels: 0,
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
  const rmsModule = device.createShaderModule({ code: createDecodeRmsSrqShader(subgroupReduction) });
  const qkvModule = device.createShaderModule({ code: createDecodeQkvPresrqShader(subgroupReduction) });
  const [rms, qkv] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Composed DecodeRmsSrq",
      layout: "auto",
      compute: { module: rmsModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: "Composed DecodeQkvProj",
      layout: "auto",
      compute: { module: qkvModule, entryPoint: "main" },
    }),
  ]);
  return { rms, qkv, compileMs: performance.now() - started };
}

function createResources(
  device: GPUDevice,
  pipelines: CompiledPipelines,
  rmsFixture: Awaited<ReturnType<typeof loadDecodeRmsFixture>>,
  qkvFixture: Awaited<ReturnType<typeof loadCapturedQatQkvFixture>>,
): ComposedResources {
  const hiddenBuffer = createBuffer(device, "Decode hidden", rmsFixture.hidden.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const rmsWeightBuffer = createBuffer(device, "Decode RMS weight", rmsFixture.weight.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const rmsOutputBuffer = createBuffer(device, "Decode RMS output QKV input", rmsFixture.expectedOutput.byteLength, GPUBufferUsage.STORAGE);
  const rmsSumBuffer = createBuffer(device, "Decode RMS sum QKV input", 4, GPUBufferUsage.STORAGE);
  const qkvWeightBuffer = createBuffer(device, "Decode QKV weights", qkvFixture.packedWeights.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const qkvScaleBuffer = createBuffer(device, "Decode QKV scales", qkvFixture.rowScales.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outputBuffer = createBuffer(device, "Decode QKV output", (Q_OUT + 2 * KV_OUT) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const rmsParamsBuffer = createBuffer(device, "Decode RMS parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const qkvParamsBuffer = createBuffer(device, "Decode QKV parameters", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const readBuffer = createBuffer(device, "Decode QKV readback", outputBuffer.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const buffers = [
    hiddenBuffer,
    rmsWeightBuffer,
    rmsOutputBuffer,
    rmsSumBuffer,
    qkvWeightBuffer,
    qkvScaleBuffer,
    outputBuffer,
    rmsParamsBuffer,
    qkvParamsBuffer,
    readBuffer,
  ];

  const rmsParams = new ArrayBuffer(16);
  const rmsParamsView = new DataView(rmsParams);
  rmsParamsView.setUint32(0, 1, true);
  rmsParamsView.setUint32(4, 1, true);
  rmsParamsView.setFloat32(8, rmsFixture.inputScale, true);
  device.queue.writeBuffer(hiddenBuffer, 0, rmsFixture.hidden);
  device.queue.writeBuffer(rmsWeightBuffer, 0, rmsFixture.weight);
  device.queue.writeBuffer(qkvWeightBuffer, 0, qkvFixture.packedWeights);
  device.queue.writeBuffer(qkvScaleBuffer, 0, qkvFixture.rowScales);
  device.queue.writeBuffer(rmsParamsBuffer, 0, rmsParams);
  device.queue.writeBuffer(qkvParamsBuffer, 0, qkvFixture.outputScales);

  return {
    rmsBindGroup: device.createBindGroup({
      layout: pipelines.rms.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: rmsWeightBuffer } },
        { binding: 2, resource: { buffer: rmsOutputBuffer } },
        { binding: 3, resource: { buffer: rmsSumBuffer } },
        { binding: 4, resource: { buffer: rmsParamsBuffer } },
      ],
    }),
    qkvBindGroup: device.createBindGroup({
      layout: pipelines.qkv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: rmsOutputBuffer } },
        { binding: 1, resource: { buffer: qkvWeightBuffer } },
        { binding: 2, resource: { buffer: qkvScaleBuffer } },
        { binding: 3, resource: { buffer: rmsSumBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
        { binding: 5, resource: { buffer: qkvParamsBuffer } },
      ],
    }),
    outputBuffer,
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
    rmsBeginning?: number;
    qkvEnd?: number;
  },
): void {
  const rmsPass = encoder.beginComputePass({
    label: "Composed DecodeRmsSrq",
    timestampWrites: timestampWrites?.rmsBeginning === undefined ? undefined : {
      querySet: timestampWrites.querySet,
      beginningOfPassWriteIndex: timestampWrites.rmsBeginning,
    },
  });
  rmsPass.setPipeline(pipelines.rms);
  rmsPass.setBindGroup(0, resources.rmsBindGroup);
  rmsPass.dispatchWorkgroups(1);
  rmsPass.end();

  const qkvPass = encoder.beginComputePass({
    label: "Composed DecodeQkvProj",
    timestampWrites: timestampWrites?.qkvEnd === undefined ? undefined : {
      querySet: timestampWrites.querySet,
      endOfPassWriteIndex: timestampWrites.qkvEnd,
    },
  });
  qkvPass.setPipeline(pipelines.qkv);
  qkvPass.setBindGroup(0, resources.qkvBindGroup);
  qkvPass.dispatchWorkgroups(QKV_WORKGROUP_COUNT);
  qkvPass.end();
}

async function dispatch(
  device: GPUDevice,
  pipelines: CompiledPipelines,
  resources: ComposedResources,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "Composed decode dispatch" });
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
    label: "Composed decode timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "Composed decode timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < 10; sample += 1) {
      const encoder = device.createCommandEncoder({ label: "Composed decode timestamp sample" });
      for (let pair = 0; pair < DISPATCHES_PER_TIMESTAMP_SAMPLE; pair += 1) {
        encodePair(encoder, pipelines, resources, {
          querySet,
          rmsBeginning: pair === 0 ? 0 : undefined,
          qkvEnd: pair === DISPATCHES_PER_TIMESTAMP_SAMPLE - 1 ? 1 : undefined,
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

function assertExactBoundary(actual: Float32Array, expected: Float32Array, name: string): void {
  if (actual.length !== expected.length) {
    throw new Error(`Decode RMS/QKV ${name} boundary length mismatch`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(`Decode RMS/QKV ${name} boundary mismatch at ${index}`);
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