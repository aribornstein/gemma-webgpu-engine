import type { MaterializedGemmaOutputWeights } from "../model/gemma-output-weights";

const HIDDEN_SIZE = 1536;
const WORDS_PER_ROW = HIDDEN_SIZE / 16;
const ROWS_PER_SUBGROUP = 4;
const SUBGROUPS_PER_WORKGROUP = 2;
const ROWS_PER_WORKGROUP = ROWS_PER_SUBGROUP * SUBGROUPS_PER_WORKGROUP;
const VOCAB_SIZE = 262144;

export type GemmaLmHeadMode = "row-major-subgroups" | "block-major-columns";

export interface GemmaLmHeadPipeline {
  pipeline: GPUComputePipeline;
  outputFeatures: number;
  workgroupCount: number;
  mode: GemmaLmHeadMode;
}

export interface GemmaLmHeadInputs {
  activation: GPUBuffer;
  activationSum: GPUBuffer;
}

export interface GemmaLmHeadResources {
  bindGroup: GPUBindGroup;
  logits: GPUBuffer;
  modelWeights: {
    packed: GPUBuffer;
    rowScales: GPUBuffer;
    inputScale: number;
    outputScale: number;
  };
  buffers: GPUBuffer[];
  bytesAllocated: number;
}

export interface GemmaLmHeadWeights {
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  inputScale?: number;
  outputScale: number;
}

const pipelineCache = new WeakMap<GPUDevice, Map<string, Promise<GemmaLmHeadPipeline>>>();

export function getGemmaLmHeadPipeline(
  device: GPUDevice,
  outputFeatures = VOCAB_SIZE,
  mode: GemmaLmHeadMode = "block-major-columns",
): Promise<GemmaLmHeadPipeline> {
  validateOutputFeatures(outputFeatures);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cacheKey = `${outputFeatures}:${mode}`;
  const cached = devicePipelines.get(cacheKey);
  if (cached) return cached;
  const compiled = compileGemmaLmHeadPipeline(device, outputFeatures, mode).catch((error) => {
    devicePipelines?.delete(cacheKey);
    throw error;
  });
  devicePipelines.set(cacheKey, compiled);
  return compiled;
}

export async function compileGemmaLmHeadPipeline(
  device: GPUDevice,
  outputFeatures = VOCAB_SIZE,
  mode: GemmaLmHeadMode = "block-major-columns",
): Promise<GemmaLmHeadPipeline> {
  validateOutputFeatures(outputFeatures);
  const module = device.createShaderModule({
    code: mode === "block-major-columns"
      ? createGemmaLmHeadBlockMajorShader(outputFeatures)
      : createGemmaLmHeadShader(outputFeatures),
  });
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma int2 LM head ${mode} (${outputFeatures} rows)`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return {
    pipeline,
    outputFeatures,
    workgroupCount: Math.ceil(
      outputFeatures / (mode === "block-major-columns" ? 128 : ROWS_PER_WORKGROUP),
    ),
    mode,
  };
}

export function createGemmaLmHeadResources(
  device: GPUDevice,
  pipeline: GemmaLmHeadPipeline,
  inputs: GemmaLmHeadInputs,
  weights: GemmaLmHeadWeights | MaterializedGemmaOutputWeights,
): GemmaLmHeadResources {
  const expectedWords = pipeline.outputFeatures * WORDS_PER_ROW;
  if (weights.packedWeights.length !== expectedWords ||
      weights.rowScales.length !== pipeline.outputFeatures) {
    throw new Error(
      `Gemma LM head weights do not match ${pipeline.outputFeatures} output features`,
    );
  }
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) =>
    device.createBuffer({ label, size, usage });
  const weightBuffer = make(
    "Gemma LM head int2 weights",
    weights.packedWeights.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const scaleBuffer = make(
    "Gemma LM head row scales",
    weights.rowScales.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  );
  const logits = make(
    "Gemma LM head logits",
    pipeline.outputFeatures * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const paramsBuffer = make(
    "Gemma LM head parameters",
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  const buffers = [weightBuffer, scaleBuffer, logits, paramsBuffer];
  device.queue.writeBuffer(
    weightBuffer,
    0,
    pipeline.mode === "block-major-columns"
      ? repackLmHeadWeightsBlockMajor(weights.packedWeights, pipeline.outputFeatures)
      : weights.packedWeights,
  );
  device.queue.writeBuffer(scaleBuffer, 0, weights.rowScales);
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([weights.outputScale, 0, 0, 0]));
  return {
    bindGroup: device.createBindGroup({
      layout: pipeline.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputs.activation } },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: scaleBuffer } },
        { binding: 3, resource: { buffer: inputs.activationSum } },
        { binding: 4, resource: { buffer: logits } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    }),
    logits,
    modelWeights: {
      packed: weightBuffer,
      rowScales: scaleBuffer,
      inputScale: weights.inputScale ?? 0,
      outputScale: weights.outputScale,
    },
    buffers,
    bytesAllocated: buffers.reduce((total, buffer) => total + buffer.size, 0),
  };
}

export function encodeGemmaLmHead(
  encoder: GPUCommandEncoder,
  pipeline: GemmaLmHeadPipeline,
  resources: GemmaLmHeadResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma int2 LM head" });
  encodeGemmaLmHeadPass(pass, pipeline, resources);
  pass.end();
}

export function encodeGemmaLmHeadPass(
  pass: GPUComputePassEncoder,
  pipeline: GemmaLmHeadPipeline,
  resources: GemmaLmHeadResources,
): void {
  pass.setPipeline(pipeline.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(pipeline.workgroupCount);
}

export function destroyGemmaLmHeadResources(resources: GemmaLmHeadResources): void {
  for (const buffer of resources.buffers) buffer.destroy();
}

export function createGemmaLmHeadShader(outputFeatures = VOCAB_SIZE): string {
  validateOutputFeatures(outputFeatures);
  return `enable subgroups;

struct Params { outputScale: f32 }
@group(0) @binding(0) var<storage, read> activation: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> sumA: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const OUT_FEATURES: u32 = ${outputFeatures}u;
const WORDS_PER_ROW: u32 = ${WORDS_PER_ROW}u;
const ROWS_PER_SUBGROUP: u32 = ${ROWS_PER_SUBGROUP}u;
const ROWS_PER_WORKGROUP: u32 = ${ROWS_PER_WORKGROUP}u;
const ZERO_POINT: f32 = 2.0;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
) {
  let subgroupId = localId.x >> 5u;
  let lane = localId.x & 31u;
  let rowBase = workgroupId.x * ROWS_PER_WORKGROUP +
    subgroupId * ROWS_PER_SUBGROUP;
  var accumulators: array<f32, ${ROWS_PER_SUBGROUP}>;
  for (var row = 0u; row < ROWS_PER_SUBGROUP; row = row + 1u) {
    accumulators[row] = 0.0;
  }

  var word = lane;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
    var values: array<vec4<f32>, 4>;
    for (var chunk = 0u; chunk < 4u; chunk = chunk + 1u) {
      values[chunk] = activation[word * 4u + chunk];
    }
    for (var row = 0u; row < ROWS_PER_SUBGROUP; row = row + 1u) {
      let outputRow = rowBase + row;
      if (outputRow < OUT_FEATURES) {
        let packed = bits[outputRow * WORDS_PER_ROW + word];
        let code0 = unpack4x8unorm(packed & 0x03030303u);
        let code1 = unpack4x8unorm((packed >> 2u) & 0x03030303u);
        let code2 = unpack4x8unorm((packed >> 4u) & 0x03030303u);
        let code3 = unpack4x8unorm((packed >> 6u) & 0x03030303u);
        accumulators[row] = accumulators[row] +
          ((dot(vec4<f32>(code0.x, code1.x, code2.x, code3.x), values[0]) +
          dot(vec4<f32>(code0.y, code1.y, code2.y, code3.y), values[1])) +
          (dot(vec4<f32>(code0.z, code1.z, code2.z, code3.z), values[2]) +
          dot(vec4<f32>(code0.w, code1.w, code2.w, code3.w), values[3])));
      }
    }
    word = word + 32u;
  }

  for (var row = 0u; row < ROWS_PER_SUBGROUP; row = row + 1u) {
    let outputRow = rowBase + row;
    let sum = subgroupAdd(accumulators[row]);
    if (lane == 0u && outputRow < OUT_FEATURES) {
      logits[outputRow] = srq(
        scales[outputRow] * fma(sum, 255.0, -(ZERO_POINT * sumA[0])),
        params.outputScale,
      );
    }
  }
}`;
}

export function createGemmaLmHeadBlockMajorShader(outputFeatures = VOCAB_SIZE): string {
  validateOutputFeatures(outputFeatures);
  return `
struct Params { outputScale: f32 }
@group(0) @binding(0) var<storage, read> activation: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bits: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> sumA: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const OUT_FEATURES: u32 = ${outputFeatures}u;
const BLOCKS_PER_ROW: u32 = ${WORDS_PER_ROW / 4}u;
const ZERO_POINT: f32 = 2.0;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

fn blockDot(packed: vec4<u32>, activationBase: u32) -> f32 {
  let code00 = unpack4x8unorm(packed[0] & 0x03030303u);
  let code01 = unpack4x8unorm((packed[0] >> 2u) & 0x03030303u);
  let code02 = unpack4x8unorm((packed[0] >> 4u) & 0x03030303u);
  let code03 = unpack4x8unorm((packed[0] >> 6u) & 0x03030303u);
  let code10 = unpack4x8unorm(packed[1] & 0x03030303u);
  let code11 = unpack4x8unorm((packed[1] >> 2u) & 0x03030303u);
  let code12 = unpack4x8unorm((packed[1] >> 4u) & 0x03030303u);
  let code13 = unpack4x8unorm((packed[1] >> 6u) & 0x03030303u);
  let code20 = unpack4x8unorm(packed[2] & 0x03030303u);
  let code21 = unpack4x8unorm((packed[2] >> 2u) & 0x03030303u);
  let code22 = unpack4x8unorm((packed[2] >> 4u) & 0x03030303u);
  let code23 = unpack4x8unorm((packed[2] >> 6u) & 0x03030303u);
  let code30 = unpack4x8unorm(packed[3] & 0x03030303u);
  let code31 = unpack4x8unorm((packed[3] >> 2u) & 0x03030303u);
  let code32 = unpack4x8unorm((packed[3] >> 4u) & 0x03030303u);
  let code33 = unpack4x8unorm((packed[3] >> 6u) & 0x03030303u);
  let sum0 =
    (dot(vec4<f32>(code00.x, code01.x, code02.x, code03.x), activation[activationBase + 0u]) +
    dot(vec4<f32>(code00.y, code01.y, code02.y, code03.y), activation[activationBase + 1u])) +
    (dot(vec4<f32>(code00.z, code01.z, code02.z, code03.z), activation[activationBase + 2u]) +
    dot(vec4<f32>(code00.w, code01.w, code02.w, code03.w), activation[activationBase + 3u]));
  let sum1 =
    (dot(vec4<f32>(code10.x, code11.x, code12.x, code13.x), activation[activationBase + 4u]) +
    dot(vec4<f32>(code10.y, code11.y, code12.y, code13.y), activation[activationBase + 5u])) +
    (dot(vec4<f32>(code10.z, code11.z, code12.z, code13.z), activation[activationBase + 6u]) +
    dot(vec4<f32>(code10.w, code11.w, code12.w, code13.w), activation[activationBase + 7u]));
  let sum2 =
    (dot(vec4<f32>(code20.x, code21.x, code22.x, code23.x), activation[activationBase + 8u]) +
    dot(vec4<f32>(code20.y, code21.y, code22.y, code23.y), activation[activationBase + 9u])) +
    (dot(vec4<f32>(code20.z, code21.z, code22.z, code23.z), activation[activationBase + 10u]) +
    dot(vec4<f32>(code20.w, code21.w, code22.w, code23.w), activation[activationBase + 11u]));
  let sum3 =
    (dot(vec4<f32>(code30.x, code31.x, code32.x, code33.x), activation[activationBase + 12u]) +
    dot(vec4<f32>(code30.y, code31.y, code32.y, code33.y), activation[activationBase + 13u])) +
    (dot(vec4<f32>(code30.z, code31.z, code32.z, code33.z), activation[activationBase + 14u]) +
    dot(vec4<f32>(code30.w, code31.w, code32.w, code33.w), activation[activationBase + 15u]));
  return (sum0 + sum1) + (sum2 + sum3);
}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
) {
  let outputRow = workgroupId.x * 128u + localId.x;
  if (outputRow >= OUT_FEATURES) { return; }
  var accumulator = 0.0;
  for (var block = 0u; block < BLOCKS_PER_ROW; block = block + 1u) {
    accumulator = accumulator + blockDot(
      bits[block * OUT_FEATURES + outputRow],
      block * 16u,
    );
  }
  logits[outputRow] = srq(
    scales[outputRow] * fma(accumulator, 255.0, -(ZERO_POINT * sumA[0])),
    params.outputScale,
  );
}`;
}

function repackLmHeadWeightsBlockMajor(
  rowMajor: Uint32Array,
  outputFeatures: number,
): Uint32Array {
  const blockMajor = new Uint32Array(rowMajor.length);
  const blocksPerRow = WORDS_PER_ROW / 4;
  for (let row = 0; row < outputFeatures; row += 1) {
    for (let block = 0; block < blocksPerRow; block += 1) {
      const source = row * WORDS_PER_ROW + block * 4;
      const destination = (block * outputFeatures + row) * 4;
      blockMajor[destination] = rowMajor[source];
      blockMajor[destination + 1] = rowMajor[source + 1];
      blockMajor[destination + 2] = rowMajor[source + 2];
      blockMajor[destination + 3] = rowMajor[source + 3];
    }
  }
  return blockMajor;
}

function validateOutputFeatures(outputFeatures: number): void {
  if (!Number.isInteger(outputFeatures) || outputFeatures < 1) {
    throw new Error("Gemma LM head outputFeatures must be a positive integer");
  }
}