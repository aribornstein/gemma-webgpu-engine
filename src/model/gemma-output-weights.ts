import type {
  CachedTensorDescriptor,
  CachedTensorPayload,
} from "./cached-safetensors";
import {
  bfloat16ToFloat32,
  float32LittleEndian,
  packedUint32,
} from "./gemma-layer-materializer";
import type { GemmaLayerTensorSource } from "./gemma-layer-weights";

const VOCAB_SIZE = 262144;
const HIDDEN_SIZE = 1536;
const FINAL_NORM = "model.language_model.norm.weight";
const LM_HEAD = "lm_head";

export const GEMMA_OUTPUT_TENSOR_NAMES = [
  FINAL_NORM,
  `${LM_HEAD}.input_activation_scale`,
  `${LM_HEAD}.output_activation_scale`,
  `${LM_HEAD}.weight_scale`,
  `${LM_HEAD}.weight`,
] as const;

export interface MaterializedGemmaOutputWeights {
  finalNorm: Float32Array;
  inputScale: number;
  outputScale: number;
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  sourceBytes: number;
}

export async function loadGemmaOutputWeights(
  source: GemmaLayerTensorSource,
): Promise<MaterializedGemmaOutputWeights> {
  const expected = outputContracts();
  for (const contract of expected) {
    const descriptor = source.descriptors.get(contract.name);
    if (!descriptor || !matchesContract(descriptor, contract)) {
      throw new Error(`Gemma output tensor contract mismatch for ${contract.name}`);
    }
  }
  const tensors = await source.readTensors(GEMMA_OUTPUT_TENSOR_NAMES);
  const payload = (name: string): CachedTensorPayload => {
    const tensor = tensors.get(name);
    const descriptor = source.descriptors.get(name);
    if (!tensor || !descriptor || tensor.byteLength !== descriptor.byteLength ||
        tensor.bytes.byteLength !== descriptor.byteLength || tensor.sha256.length !== 64) {
      throw new Error(`Gemma output load omitted or corrupted tensor ${name}`);
    }
    return tensor;
  };
  const scalar = (name: string): number => {
    const values = float32LittleEndian(payload(name));
    if (values.length !== 1) throw new Error(`Gemma output tensor ${name} is not scalar`);
    return values[0];
  };
  return {
    finalNorm: bfloat16ToFloat32(payload(FINAL_NORM)),
    inputScale: scalar(`${LM_HEAD}.input_activation_scale`),
    outputScale: scalar(`${LM_HEAD}.output_activation_scale`),
    rowScales: float32LittleEndian(payload(`${LM_HEAD}.weight_scale`)),
    packedWeights: packedUint32(payload(`${LM_HEAD}.weight`)),
    sourceBytes: GEMMA_OUTPUT_TENSOR_NAMES.reduce(
      (total, name) => total + payload(name).byteLength,
      0,
    ),
  };
}

function outputContracts(): CachedTensorDescriptor[] {
  return [
    contract(FINAL_NORM, "BF16", [HIDDEN_SIZE], HIDDEN_SIZE * 2),
    contract(`${LM_HEAD}.input_activation_scale`, "F32", [], 4),
    contract(`${LM_HEAD}.output_activation_scale`, "F32", [], 4),
    contract(`${LM_HEAD}.weight_scale`, "F32", [VOCAB_SIZE, 1], VOCAB_SIZE * 4),
    contract(`${LM_HEAD}.weight`, "U8", [VOCAB_SIZE, HIDDEN_SIZE / 4],
      VOCAB_SIZE * HIDDEN_SIZE / 4),
  ];
}

function contract(
  name: string,
  dtype: string,
  shape: readonly number[],
  byteLength: number,
): CachedTensorDescriptor {
  return { name, dtype, shape, begin: 0, end: byteLength, byteLength };
}

function matchesContract(
  descriptor: CachedTensorDescriptor,
  expected: CachedTensorDescriptor,
): boolean {
  return descriptor.dtype === expected.dtype &&
    descriptor.byteLength === expected.byteLength &&
    descriptor.end - descriptor.begin === descriptor.byteLength &&
    descriptor.shape.length === expected.shape.length &&
    descriptor.shape.every((value, index) => value === expected.shape[index]);
}