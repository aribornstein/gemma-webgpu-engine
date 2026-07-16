import type { GemmaInputWeights, GemmaTokenInputs } from "../model/gemma-input-weights";

const HIDDEN_SIZE = 1536;
const LAYER_COUNT = 35;
const PER_LAYER_SIZE = 256;
const PER_LAYER_TOTAL = LAYER_COUNT * PER_LAYER_SIZE;

export interface GemmaDecodeInputPipeline {
  projection: GPUComputePipeline;
  rms: GPUComputePipeline;
  add: GPUComputePipeline;
  scale: GPUComputePipeline;
}

export interface GemmaDecodeInputModelWeightBuffers {
  projection: GPUBuffer;
  projectionNorm: GPUBuffer;
}

export interface GemmaDecodeInputResources {
  projectionBindGroup: GPUBindGroup;
  rmsBindGroup: GPUBindGroup;
  addBindGroup: GPUBindGroup;
  scaleBindGroup: GPUBindGroup;
  hidden: GPUBuffer;
  perLayerInputs: GPUBuffer;
  hiddenUpload: GPUBuffer;
  perLayerEmbeddingUpload: GPUBuffer;
  modelWeights: GemmaDecodeInputModelWeightBuffers;
  rowCount: number;
  buffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<GPUDevice, Promise<GemmaDecodeInputPipeline>>();

export function getGemmaDecodeInputPipeline(
  device: GPUDevice,
): Promise<GemmaDecodeInputPipeline> {
  const cached = pipelineCache.get(device);
  if (cached) return cached;
  const compiled = compileGemmaDecodeInputPipeline(device).catch((error) => {
    pipelineCache.delete(device);
    throw error;
  });
  pipelineCache.set(device, compiled);
  return compiled;
}

export async function compileGemmaDecodeInputPipeline(
  device: GPUDevice,
): Promise<GemmaDecodeInputPipeline> {
  const module = device.createShaderModule({ code: createGemmaDecodeInputShader() });
  const [projection, rms, add, scale] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma initial per-layer projection",
      layout: "auto",
      compute: { module, entryPoint: "project" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma initial per-layer exact RMS",
      layout: "auto",
      compute: { module, entryPoint: "rms" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma initial per-layer embedding add",
      layout: "auto",
      compute: { module, entryPoint: "add" },
    }),
    device.createComputePipelineAsync({
      label: "Gemma initial per-layer scale",
      layout: "auto",
      compute: { module, entryPoint: "scale" },
    }),
  ]);
  return { projection, rms, add, scale };
}

export function createGemmaDecodeInputResources(
  device: GPUDevice,
  pipelines: GemmaDecodeInputPipeline,
  weights: GemmaInputWeights | null,
  rowCount = 1,
  sharedWeights?: GemmaDecodeInputModelWeightBuffers,
): GemmaDecodeInputResources {
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error("Gemma input row count must be a positive integer");
  }
  if ((!weights && !sharedWeights) ||
      (weights && (weights.projectionBfloat16.length !== PER_LAYER_TOTAL * HIDDEN_SIZE / 2 ||
      weights.projectionNorm.length !== PER_LAYER_SIZE))) {
    throw new Error("Gemma input weights do not match the model geometry");
  }
  const projectionBytes = PER_LAYER_TOTAL * HIDDEN_SIZE * 2;
  const projectionNormBytes = PER_LAYER_SIZE * 4;
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const hiddenUpload = make("Gemma token hidden upload", rowCount * HIDDEN_SIZE * 4, storageUpload);
  const perLayerEmbeddingUpload = make(
    "Gemma token per-layer embedding upload",
    rowCount * PER_LAYER_TOTAL * 4,
    storageUpload,
  );
  const projectionWeights = sharedWeights?.projection ?? make(
    "Gemma per-layer model projection BF16",
    projectionBytes,
    storageUpload,
  );
  const projectionNorm = sharedWeights?.projectionNorm ?? make(
    "Gemma per-layer projection norm",
    projectionNormBytes,
    storageUpload,
  );
  if (projectionWeights.size < projectionBytes || projectionNorm.size < projectionNormBytes) {
    throw new Error("Shared Gemma input weights do not match the model geometry");
  }
  const projected = make(
    "Gemma projected per-layer input",
    rowCount * PER_LAYER_TOTAL * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const hidden = make(
    "Gemma decode hidden",
    rowCount * HIDDEN_SIZE * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const perLayerInputs = make(
    "Gemma decode per-layer inputs",
    rowCount * PER_LAYER_TOTAL * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const buffers = [
    hiddenUpload,
    perLayerEmbeddingUpload,
    ...(sharedWeights ? [] : [projectionWeights, projectionNorm]),
    projected,
    hidden,
    perLayerInputs,
  ];
  if (!sharedWeights) {
    device.queue.writeBuffer(projectionWeights, 0, weights!.projectionBfloat16);
    device.queue.writeBuffer(projectionNorm, 0, weights!.projectionNorm);
  }
  const entry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({
    binding,
    resource: { buffer },
  });
  return {
    projectionBindGroup: device.createBindGroup({
      layout: pipelines.projection.getBindGroupLayout(0),
      entries: [entry(0, hiddenUpload), entry(1, projectionWeights), entry(2, projected), entry(5, hidden)],
    }),
    rmsBindGroup: device.createBindGroup({
      layout: pipelines.rms.getBindGroupLayout(0),
      entries: [
        entry(2, projected),
        entry(3, projectionNorm),
        entry(6, perLayerInputs),
      ],
    }),
    addBindGroup: device.createBindGroup({
      layout: pipelines.add.getBindGroupLayout(0),
      entries: [entry(4, perLayerEmbeddingUpload), entry(6, perLayerInputs)],
    }),
    scaleBindGroup: device.createBindGroup({
      layout: pipelines.scale.getBindGroupLayout(0),
      entries: [entry(6, perLayerInputs)],
    }),
    hidden,
    perLayerInputs,
    hiddenUpload,
    perLayerEmbeddingUpload,
    modelWeights: {
      projection: projectionWeights,
      projectionNorm,
    },
    rowCount,
    buffers,
  };
}

export function uploadGemmaTokenInputs(
  device: GPUDevice,
  resources: GemmaDecodeInputResources,
  inputs: GemmaTokenInputs,
): void {
  if (resources.rowCount !== 1) {
    throw new Error("Gemma single-token upload requires one input row");
  }
  uploadGemmaTokenInputBatch(device, resources, [inputs]);
}

export function uploadGemmaTokenInputBatch(
  device: GPUDevice,
  resources: GemmaDecodeInputResources,
  inputs: readonly GemmaTokenInputs[],
): void {
  if (inputs.length !== resources.rowCount) {
    throw new Error(
      `Gemma input batch has ${inputs.length} rows; expected ${resources.rowCount}`,
    );
  }
  const hidden = new Float32Array(resources.rowCount * HIDDEN_SIZE);
  const perLayerEmbedding = new Float32Array(resources.rowCount * PER_LAYER_TOTAL);
  for (let row = 0; row < inputs.length; row += 1) {
    const input = inputs[row];
    if (input.hidden.length !== HIDDEN_SIZE ||
        input.perLayerEmbedding.length !== PER_LAYER_TOTAL) {
      throw new Error("Gemma token inputs do not match model geometry");
    }
    hidden.set(input.hidden, row * HIDDEN_SIZE);
    perLayerEmbedding.set(input.perLayerEmbedding, row * PER_LAYER_TOTAL);
  }
  device.queue.writeBuffer(resources.hiddenUpload, 0, hidden);
  device.queue.writeBuffer(resources.perLayerEmbeddingUpload, 0, perLayerEmbedding);
}

export function encodeGemmaDecodeInput(
  encoder: GPUCommandEncoder,
  pipelines: GemmaDecodeInputPipeline,
  resources: GemmaDecodeInputResources,
): void {
  const projection = encoder.beginComputePass({ label: "Gemma initial per-layer projection" });
  projection.setPipeline(pipelines.projection);
  projection.setBindGroup(0, resources.projectionBindGroup);
  projection.dispatchWorkgroups(PER_LAYER_TOTAL / 8, resources.rowCount);
  projection.end();
  const rms = encoder.beginComputePass({ label: "Gemma initial per-layer exact RMS" });
  rms.setPipeline(pipelines.rms);
  rms.setBindGroup(0, resources.rmsBindGroup);
  rms.dispatchWorkgroups(resources.rowCount);
  rms.end();
  const add = encoder.beginComputePass({ label: "Gemma initial per-layer embedding add" });
  add.setPipeline(pipelines.add);
  add.setBindGroup(0, resources.addBindGroup);
  add.dispatchWorkgroups(LAYER_COUNT, resources.rowCount);
  add.end();
  const scale = encoder.beginComputePass({ label: "Gemma initial per-layer scale" });
  scale.setPipeline(pipelines.scale);
  scale.setBindGroup(0, resources.scaleBindGroup);
  scale.dispatchWorkgroups(LAYER_COUNT, resources.rowCount);
  scale.end();
}

export function destroyGemmaDecodeInputResources(resources: GemmaDecodeInputResources): void {
  for (const buffer of resources.buffers) buffer.destroy();
}

export function createGemmaDecodeInputShader(): string {
  return `enable subgroups;

@group(0) @binding(0) var<storage, read> tokenHidden: array<f32>;
@group(0) @binding(1) var<storage, read> projectionWeights: array<u32>;
@group(0) @binding(2) var<storage, read_write> projected: array<f32>;
@group(0) @binding(3) var<storage, read> projectionNorm: array<f32>;
@group(0) @binding(4) var<storage, read> perLayerEmbedding: array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(6) var<storage, read_write> perLayerInputs: array<f32>;

var<workgroup> partial: array<f32, 256>;

fn bfloat16(value: u32) -> f32 {
  return bitcast<f32>(value << 16u);
}

fn reduceSum(value: f32, lane: u32) -> f32 {
  partial[lane] = value;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (lane < stride) { partial[lane] = partial[lane] + partial[lane + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride = stride >> 1u;
  }
  return partial[0];
}

@compute @workgroup_size(32, 1, 1)
fn project(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let rowBase = workgroupId.x * 8u;
  let hiddenBase = workgroupId.y * ${HIDDEN_SIZE}u;
  let projectedBase = workgroupId.y * ${PER_LAYER_TOTAL}u;
  var accumulators: array<f32, 8>;
  for (var row = 0u; row < 8u; row = row + 1u) {
    accumulators[row] = 0.0;
  }
  var column4 = lane;
  loop {
    if (column4 >= ${HIDDEN_SIZE / 4}u) { break; }
    let column = column4 * 4u;
    let activation = vec4<f32>(
      tokenHidden[hiddenBase + column],
      tokenHidden[hiddenBase + column + 1u],
      tokenHidden[hiddenBase + column + 2u],
      tokenHidden[hiddenBase + column + 3u],
    );
    for (var row = 0u; row < 8u; row = row + 1u) {
      let weightBase = (rowBase + row) * ${HIDDEN_SIZE / 2}u + (column >> 1u);
      let packed01 = projectionWeights[weightBase];
      let packed23 = projectionWeights[weightBase + 1u];
      let weights = vec4<f32>(
        bfloat16(packed01 & 0xffffu),
        bfloat16(packed01 >> 16u),
        bfloat16(packed23 & 0xffffu),
        bfloat16(packed23 >> 16u),
      );
      accumulators[row] = accumulators[row] + dot(weights, activation);
    }
    column4 = column4 + 32u;
  }
  for (var row = 0u; row < 8u; row = row + 1u) {
    let total = subgroupAdd(accumulators[row]);
    if (lane == 0u) {
      projected[projectedBase + rowBase + row] = total * ${1 / Math.sqrt(HIDDEN_SIZE)};
    }
  }
  if (workgroupId.x == 0u) {
    var index = lane;
    loop {
      if (index >= ${HIDDEN_SIZE}u) { break; }
      hidden[hiddenBase + index] = tokenHidden[hiddenBase + index];
      index = index + 32u;
    }
  }
}

fn divExact(value: f32, divisor: f32, reciprocal: f32) -> f32 {
  let quotient = value * reciprocal;
  let remainder = fma(-divisor, quotient, value);
  return fma(remainder, reciprocal, quotient);
}

fn reciprocalExact(value: f32) -> f32 {
  let first = 1.0 / value;
  let second = fma(fma(-value, first, 1.0), first, first);
  return fma(fma(-value, second, 1.0), second, second);
}

fn sqrtExact(value: f32) -> f32 {
  var inverse = inverseSqrt(value);
  var root = value * inverse;
  var halfInverse = 0.5 * inverse;
  inverse = fma(-root, halfInverse, 0.5);
  root = fma(root, inverse, root);
  halfInverse = fma(halfInverse, inverse, halfInverse);
  inverse = fma(-root, halfInverse, 1.5);
  halfInverse = halfInverse + halfInverse;
  halfInverse = halfInverse * inverse;
  root = halfInverse * value;
  inverse = fma(halfInverse, value, -root);
  var correction = fma(-halfInverse, root, 1.0);
  correction = fma(-halfInverse, inverse, correction);
  halfInverse = 0.5 * root;
  halfInverse = fma(halfInverse, correction, inverse);
  return halfInverse + root;
}

fn projectedVec4(base: u32, vector: u32) -> vec4<f32> {
  let index = base + vector * 4u;
  return vec4<f32>(
    projected[index],
    projected[index + 1u],
    projected[index + 2u],
    projected[index + 3u],
  );
}

@compute @workgroup_size(64, 1, 1)
fn rms(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  if (lane >= ${LAYER_COUNT}u) { return; }
  let base = workgroupId.x * ${PER_LAYER_TOTAL}u + lane * ${PER_LAYER_SIZE}u;
  var stream0 = vec4<f32>(0.0);
  var stream1 = vec4<f32>(0.0);
  var stream2 = vec4<f32>(0.0);
  var stream3 = vec4<f32>(0.0);
  for (var group = 0u; group < 16u; group = group + 1u) {
    let vector = group * 4u;
    let value0 = projectedVec4(base, vector);
    let value1 = projectedVec4(base, vector + 1u);
    let value2 = projectedVec4(base, vector + 2u);
    let value3 = projectedVec4(base, vector + 3u);
    stream0 = fma(fma(value0, value0, vec4<f32>(0.0)), vec4<f32>(1.0), stream0);
    stream1 = fma(fma(value1, value1, vec4<f32>(0.0)), vec4<f32>(1.0), stream1);
    stream2 = fma(fma(value2, value2, vec4<f32>(0.0)), vec4<f32>(1.0), stream2);
    stream3 = fma(fma(value3, value3, vec4<f32>(0.0)), vec4<f32>(1.0), stream3);
  }
  stream0 = fma(stream1, vec4<f32>(1.0), stream0);
  stream0 = fma(stream2, vec4<f32>(1.0), stream0);
  stream0 = fma(stream3, vec4<f32>(1.0), stream0);
  var squareSum = stream0.x;
  squareSum = fma(stream0.y, 1.0, squareSum);
  squareSum = fma(stream0.z, 1.0, squareSum);
  squareSum = fma(stream0.w, 1.0, squareSum);
  let meanSquare = divExact(squareSum, 256.0, 0.00390625) + 0.000001;
  let inverseRms = reciprocalExact(sqrtExact(meanSquare));
  for (var vector = 0u; vector < 64u; vector = vector + 1u) {
    let input = projectedVec4(base, vector);
    let normalized = input * vec4<f32>(inverseRms);
    let weightIndex = vector * 4u;
    let weights = vec4<f32>(
      projectionNorm[weightIndex],
      projectionNorm[weightIndex + 1u],
      projectionNorm[weightIndex + 2u],
      projectionNorm[weightIndex + 3u],
    );
    let output = normalized * weights;
    let outputIndex = base + weightIndex;
    perLayerInputs[outputIndex] = output.x;
    perLayerInputs[outputIndex + 1u] = output.y;
    perLayerInputs[outputIndex + 2u] = output.z;
    perLayerInputs[outputIndex + 3u] = output.w;
  }
}

@compute @workgroup_size(256, 1, 1)
fn add(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = workgroupId.y * ${PER_LAYER_TOTAL}u +
    workgroupId.x * ${PER_LAYER_SIZE}u + lane;
  perLayerInputs[index] = perLayerInputs[index] + perLayerEmbedding[index];
}

@compute @workgroup_size(256, 1, 1)
fn scale(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let index = workgroupId.y * ${PER_LAYER_TOTAL}u +
    workgroupId.x * ${PER_LAYER_SIZE}u + lane;
  perLayerInputs[index] = perLayerInputs[index] * ${Math.SQRT1_2};
}`;
}