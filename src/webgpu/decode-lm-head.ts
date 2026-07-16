import type { MaterializedGemmaOutputWeights } from "../model/gemma-output-weights";

const HIDDEN_SIZE = 1536;
const WORDS_PER_ROW = HIDDEN_SIZE / 16;
const ROWS_PER_SUBGROUP = 4;
const SUBGROUPS_PER_WORKGROUP = 2;
const ROWS_PER_WORKGROUP = ROWS_PER_SUBGROUP * SUBGROUPS_PER_WORKGROUP;
const VOCAB_SIZE = 262144;

export interface GemmaLmHeadPipeline {
  pipeline: GPUComputePipeline;
  outputFeatures: number;
  workgroupCount: number;
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

const pipelineCache = new WeakMap<GPUDevice, Map<number, Promise<GemmaLmHeadPipeline>>>();

export function getGemmaLmHeadPipeline(
  device: GPUDevice,
  outputFeatures = VOCAB_SIZE,
): Promise<GemmaLmHeadPipeline> {
  validateOutputFeatures(outputFeatures);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cached = devicePipelines.get(outputFeatures);
  if (cached) return cached;
  const compiled = compileGemmaLmHeadPipeline(device, outputFeatures).catch((error) => {
    devicePipelines?.delete(outputFeatures);
    throw error;
  });
  devicePipelines.set(outputFeatures, compiled);
  return compiled;
}

export async function compileGemmaLmHeadPipeline(
  device: GPUDevice,
  outputFeatures = VOCAB_SIZE,
): Promise<GemmaLmHeadPipeline> {
  validateOutputFeatures(outputFeatures);
  const module = device.createShaderModule({ code: createGemmaLmHeadShader(outputFeatures) });
  const pipeline = await device.createComputePipelineAsync({
    label: `Gemma int2 LM head (${outputFeatures} rows)`,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return {
    pipeline,
    outputFeatures,
    workgroupCount: Math.ceil(outputFeatures / ROWS_PER_WORKGROUP),
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
  device.queue.writeBuffer(weightBuffer, 0, weights.packedWeights);
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
  pass.setPipeline(pipeline.pipeline);
  pass.setBindGroup(0, resources.bindGroup);
  pass.dispatchWorkgroups(pipeline.workgroupCount);
  pass.end();
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

function validateOutputFeatures(outputFeatures: number): void {
  if (!Number.isInteger(outputFeatures) || outputFeatures < 1) {
    throw new Error("Gemma LM head outputFeatures must be a positive integer");
  }
}