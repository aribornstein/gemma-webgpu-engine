import {
  loadDecodeKNormRopeFixture,
  type DecodeKNormRopeFixture,
} from "../model/decode-k-norm-rope-fixture";
import { getWebGpuDevice } from "./device";

const WORKGROUP_SIZE = 128;
const DISPATCHES_PER_TIMESTAMP_SAMPLE = 100;

export interface DecodeKNormRopeBenchmarkResult {
  sourceOperator: "com.xenova.gemma4.DecodeQkNormRope";
  sourceVariant: "scalar";
  artifactSha256: string;
  sourceCaptureSha256: string;
  headDim: number;
  halfDim: number;
  heads: number;
  workgroupSize: number;
  workgroupCount: number;
  iterations: number;
  shaderCompilationMs: number;
  pipelineCacheHit: boolean;
  dispatchMedianMs: number;
  dispatchP95Ms: number;
  gpuKernelDispatchesPerSample: number | null;
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

interface DecodeKNormRopeResources {
  bindGroup: GPUBindGroup;
  outputBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

const pipelineCache = new WeakMap<GPUDevice, Promise<CompiledPipeline>>();

export async function benchmarkDecodeKNormRope(
  iterations = 20,
): Promise<DecodeKNormRopeBenchmarkResult> {
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Iterations must be a positive integer");
  }

  const fixture = await loadDecodeKNormRopeFixture();
  const device = await getWebGpuDevice();
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

    const encoder = device.createCommandEncoder({ label: "DecodeQkNormRope readback" });
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

    const outputErrors = measureErrors(actualOutput, fixture.expectedOutput);
    const sortedDispatchSamples = dispatchSamples.toSorted((left, right) => left - right);
    const gpuSamples = await measureGpuKernel(device, compiled.pipeline, resources.bindGroup);

    return {
      sourceOperator: "com.xenova.gemma4.DecodeQkNormRope",
      sourceVariant: "scalar",
      artifactSha256: fixture.artifactSha256,
      sourceCaptureSha256: fixture.sourceCaptureSha256,
      headDim: fixture.headDim,
      halfDim: fixture.halfDim,
      heads: fixture.heads,
      workgroupSize: WORKGROUP_SIZE,
      workgroupCount: 1,
      iterations,
      shaderCompilationMs: round(cached ? 0 : compiled.compileMs),
      pipelineCacheHit: Boolean(cached),
      dispatchMedianMs: round(percentile(sortedDispatchSamples, 0.5)),
      dispatchP95Ms: round(percentile(sortedDispatchSamples, 0.95)),
      gpuKernelDispatchesPerSample: gpuSamples ? DISPATCHES_PER_TIMESTAMP_SAMPLE : null,
      gpuKernelMedianMs: gpuSamples ? round(percentile(gpuSamples, 0.5)) : null,
      gpuKernelP95Ms: gpuSamples ? round(percentile(gpuSamples, 0.95)) : null,
      outputMaximumAbsoluteError: outputErrors.maximumAbsoluteError,
      outputMaximumRelativeError: outputErrors.maximumRelativeError,
      gpuBufferAllocations: resources.buffers.length,
      bytesAllocated: resources.bytesAllocated,
      allocationsPerDispatch: 0,
    };
  } finally {
    for (const buffer of resources.buffers) buffer.destroy();
  }
}

async function compilePipeline(device: GPUDevice): Promise<CompiledPipeline> {
  const started = performance.now();
  const module = device.createShaderModule({ code: createDecodeKNormRopeShader() });
  const pipeline = await device.createComputePipelineAsync({
    label: "DecodeQkNormRope scalar",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, compileMs: performance.now() - started };
}

export function createDecodeKNormRopeShader(
  headDim: 256 | 512 = 256,
): string {
  const halfDim = headDim / 2;
  return `struct Params {
  seq: u32,
  heads: u32,
  dstOffset: u32,
  padding: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> cosTbl: array<f32>;
@group(0) @binding(3) var<storage, read> sinTbl: array<f32>;
@group(0) @binding(4) var<storage, read_write> yn: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const HEAD_DIM: u32 = ${headDim}u;
const HALF_DIM: u32 = ${halfDim}u;
const WG: u32 = 128u;
const EPS: f32 = 0.000001;

var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let token = workgroup_id.x;
  let head = workgroup_id.y;
  if (token >= params.seq || head >= params.heads) { return; }
  let thread = local_id.x;
  let base = (token * params.heads + head) * HEAD_DIM;
  let output_base = params.dstOffset + base;
  let trig_base = token * HALF_DIM;

  var square_sum = 0.0;
  var dimension = thread;
  loop {
    if (dimension >= HEAD_DIM) { break; }
    let value = x[base + dimension];
    square_sum = square_sum + value * value;
    dimension = dimension + WG;
  }
  red[thread] = square_sum;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (thread < stride) {
      red[thread] = red[thread] + red[thread + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let scale = inverseSqrt(red[0] / f32(HEAD_DIM) + EPS);

  var index = thread;
  loop {
    if (index >= HALF_DIM) { break; }
    let first = x[base + index] * scale * w[index];
    let second = x[base + index + HALF_DIM] * scale * w[index + HALF_DIM];
    let cosine = cosTbl[trig_base + index];
    let sine = sinTbl[trig_base + index];
    yn[output_base + index] = first * cosine - second * sine;
    yn[output_base + index + HALF_DIM] = second * cosine + first * sine;
    index = index + WG;
  }
}`;
}

function createResources(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  fixture: DecodeKNormRopeFixture,
): DecodeKNormRopeResources {
  const inputBuffer = createBuffer(
    device,
    "DecodeQkNormRope input",
    fixture.input.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const weightBuffer = createBuffer(
    device,
    "DecodeQkNormRope weight",
    fixture.weight.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const cosineBuffer = createBuffer(
    device,
    "DecodeQkNormRope cosine",
    fixture.cosine.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const sineBuffer = createBuffer(
    device,
    "DecodeQkNormRope sine",
    fixture.sine.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const outputBuffer = createBuffer(
    device,
    "DecodeQkNormRope output",
    fixture.expectedOutput.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const readBuffer = createBuffer(
    device,
    "DecodeQkNormRope readback",
    fixture.expectedOutput.byteLength,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  );
  const paramsBuffer = createBuffer(
    device,
    "DecodeQkNormRope parameters",
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const buffers = [
    inputBuffer,
    weightBuffer,
    cosineBuffer,
    sineBuffer,
    outputBuffer,
    readBuffer,
    paramsBuffer,
  ];

  const parameterBytes = new Uint32Array([1, 1, 0, 0]);
  device.queue.writeBuffer(inputBuffer, 0, fixture.input);
  device.queue.writeBuffer(weightBuffer, 0, fixture.weight);
  device.queue.writeBuffer(cosineBuffer, 0, fixture.cosine);
  device.queue.writeBuffer(sineBuffer, 0, fixture.sine);
  device.queue.writeBuffer(paramsBuffer, 0, parameterBytes);

  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: cosineBuffer } },
        { binding: 3, resource: { buffer: sineBuffer } },
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
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "DecodeQkNormRope dispatch" });
  const pass = encoder.beginComputePass({ label: "DecodeQkNormRope scalar" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1, 1, 1);
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
    label: "DecodeQkNormRope timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: "DecodeQkNormRope timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      const encoder = device.createCommandEncoder({
        label: "DecodeQkNormRope timestamp sample",
      });
      const pass = encoder.beginComputePass({
        label: "DecodeQkNormRope timestamp batch",
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < DISPATCHES_PER_TIMESTAMP_SAMPLE; index += 1) {
        pass.dispatchWorkgroups(1, 1, 1);
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
          Number(timestamps[1] - timestamps[0]) / 1e6 / DISPATCHES_PER_TIMESTAMP_SAMPLE,
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
