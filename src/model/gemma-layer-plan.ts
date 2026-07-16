import type { CachedTensorDescriptor } from "./cached-safetensors";

const HIDDEN_SIZE = 1536;
const ATTENTION_HEADS = 8;
const KV_HEADS = 1;
const PLE_SIZE = 256;
const LAYER_COUNT = 35;
const INT2_MLP_START_LAYER = 15;
const K_NORM_LAYER_COUNT = 15;
const FIRST_KV_SHARED_LAYER = 15;
const LAST_SLIDING_KV_SOURCE_LAYER = 13;
const LAST_FULL_KV_SOURCE_LAYER = 14;
const SLIDING_WINDOW = 512;

type TensorDtype = "BF16" | "F32" | "I8" | "U8";

export type GemmaAttentionType = "sliding_attention" | "full_attention";
export type GemmaLayerProfile =
  | "sliding-int4"
  | "full-int4"
  | "sliding-int2"
  | "full-int2";

export interface GemmaTensorContract {
  name: string;
  dtype: TensorDtype;
  shape: readonly number[];
}

export interface GemmaProjectionPlan {
  weight: CachedTensorDescriptor;
  weightScale: CachedTensorDescriptor;
  inputActivationScale: CachedTensorDescriptor;
  outputActivationScale: CachedTensorDescriptor;
}

export interface GemmaLayerPlan {
  layerIndex: number;
  profile: GemmaLayerProfile;
  prefix: string;
  attention: {
    type: GemmaAttentionType;
    headDim: 256 | 512;
    rotaryDimensions: 128 | 256;
    slidingWindow: 512 | null;
    qOutFeatures: 2048 | 4096;
    kvOutFeatures: 256 | 512;
    isKvShared: boolean;
    kvSourceLayer: number;
    q: GemmaProjectionPlan;
    k: GemmaProjectionPlan | null;
    v: GemmaProjectionPlan | null;
    output: GemmaProjectionPlan;
    qNorm: CachedTensorDescriptor;
    kNorm: CachedTensorDescriptor | null;
  };
  mlp: {
    bits: 2 | 4;
    intermediateSize: 6144 | 12288;
    gate: GemmaProjectionPlan;
    up: GemmaProjectionPlan;
    down: GemmaProjectionPlan;
  };
  ple: {
    inputGate: GemmaProjectionPlan;
    projection: GemmaProjectionPlan;
  };
  norms: {
    input: CachedTensorDescriptor;
    postAttention: CachedTensorDescriptor;
    preFeedforward: CachedTensorDescriptor;
    postFeedforward: CachedTensorDescriptor;
    postPerLayerInput: CachedTensorDescriptor;
  };
  layerScalar: CachedTensorDescriptor;
  tensorNames: readonly string[];
  tensorBytes: number;
}

export function gemmaLayerTensorContracts(layerIndex: number): GemmaTensorContract[] {
  assertLayerIndex(layerIndex);
  const prefix = `model.language_model.layers.${layerIndex}`;
  const fullAttention = isFullAttentionLayer(layerIndex);
  const headDim = fullAttention ? 512 : 256;
  const qOutFeatures = ATTENTION_HEADS * headDim;
  const kvOutFeatures = KV_HEADS * headDim;
  const int2Mlp = layerIndex >= INT2_MLP_START_LAYER;
  const intermediateSize = int2Mlp ? 12288 : 6144;
  const mlpPackedInput = int2Mlp ? HIDDEN_SIZE / 4 : HIDDEN_SIZE / 2;
  const downPackedInput = intermediateSize / (int2Mlp ? 4 : 2);
  const contracts: GemmaTensorContract[] = [];

  addProjectionContracts(contracts, `${prefix}.self_attn.q_proj`, "U8", [
    qOutFeatures,
    HIDDEN_SIZE / 2,
  ], qOutFeatures);
  if (layerIndex < FIRST_KV_SHARED_LAYER) {
    addProjectionContracts(contracts, `${prefix}.self_attn.k_proj`, "U8", [
      kvOutFeatures,
      HIDDEN_SIZE / 2,
    ], kvOutFeatures);
    addProjectionContracts(contracts, `${prefix}.self_attn.v_proj`, "U8", [
      kvOutFeatures,
      HIDDEN_SIZE / 2,
    ], kvOutFeatures);
  }
  addProjectionContracts(contracts, `${prefix}.self_attn.o_proj`, "U8", [
    HIDDEN_SIZE,
    qOutFeatures / 2,
  ], HIDDEN_SIZE);
  contracts.push(contract(`${prefix}.self_attn.q_norm.weight`, "BF16", [headDim]));
  if (layerIndex < K_NORM_LAYER_COUNT) {
    contracts.push(contract(`${prefix}.self_attn.k_norm.weight`, "BF16", [headDim]));
  }

  addProjectionContracts(contracts, `${prefix}.mlp.gate_proj`, "U8", [
    intermediateSize,
    mlpPackedInput,
  ], intermediateSize);
  addProjectionContracts(contracts, `${prefix}.mlp.up_proj`, "U8", [
    intermediateSize,
    mlpPackedInput,
  ], intermediateSize);
  addProjectionContracts(contracts, `${prefix}.mlp.down_proj`, "U8", [
    HIDDEN_SIZE,
    downPackedInput,
  ], HIDDEN_SIZE);

  addProjectionContracts(contracts, `${prefix}.per_layer_input_gate`, "I8", [
    PLE_SIZE,
    HIDDEN_SIZE,
  ], PLE_SIZE);
  addProjectionContracts(contracts, `${prefix}.per_layer_projection`, "I8", [
    HIDDEN_SIZE,
    PLE_SIZE,
  ], HIDDEN_SIZE);

  contracts.push(
    contract(`${prefix}.input_layernorm.weight`, "BF16", [HIDDEN_SIZE]),
    contract(`${prefix}.post_attention_layernorm.weight`, "BF16", [HIDDEN_SIZE]),
    contract(`${prefix}.pre_feedforward_layernorm.weight`, "BF16", [HIDDEN_SIZE]),
    contract(`${prefix}.post_feedforward_layernorm.weight`, "BF16", [HIDDEN_SIZE]),
    contract(`${prefix}.post_per_layer_input_norm.weight`, "BF16", [HIDDEN_SIZE]),
    contract(`${prefix}.layer_scalar`, "BF16", [1]),
  );
  return contracts;
}

export function createGemmaLayerPlan(
  descriptors: ReadonlyMap<string, CachedTensorDescriptor>,
  layerIndex: number,
): GemmaLayerPlan {
  const contracts = gemmaLayerTensorContracts(layerIndex);
  const tensors = new Map<string, CachedTensorDescriptor>();
  for (const expected of contracts) {
    const actual = descriptors.get(expected.name);
    if (!actual) throw new Error(`Gemma layer ${layerIndex} is missing tensor ${expected.name}`);
    validateDescriptor(actual, expected, layerIndex);
    tensors.set(expected.name, actual);
  }

  const prefix = `model.language_model.layers.${layerIndex}`;
  const attentionType: GemmaAttentionType = isFullAttentionLayer(layerIndex)
    ? "full_attention"
    : "sliding_attention";
  const int2Mlp = layerIndex >= INT2_MLP_START_LAYER;
  const profile = `${attentionType === "full_attention" ? "full" : "sliding"}-${
    int2Mlp ? "int2" : "int4"
  }` as GemmaLayerProfile;
  const headDim = attentionType === "full_attention" ? 512 : 256;
  const qOutFeatures = attentionType === "full_attention" ? 4096 : 2048;
  const kvOutFeatures = attentionType === "full_attention" ? 512 : 256;
  const isKvShared = layerIndex >= FIRST_KV_SHARED_LAYER;
  const tensor = (suffix: string) => requiredTensor(tensors, `${prefix}.${suffix}`);
  const projection = (suffix: string): GemmaProjectionPlan => ({
    weight: tensor(`${suffix}.weight`),
    weightScale: tensor(`${suffix}.weight_scale`),
    inputActivationScale: tensor(`${suffix}.input_activation_scale`),
    outputActivationScale: tensor(`${suffix}.output_activation_scale`),
  });

  return {
    layerIndex,
    profile,
    prefix,
    attention: {
      type: attentionType,
      headDim,
      rotaryDimensions: attentionType === "full_attention" ? 128 : 256,
      slidingWindow: attentionType === "sliding_attention" ? SLIDING_WINDOW : null,
      qOutFeatures,
      kvOutFeatures,
      isKvShared,
      kvSourceLayer: isKvShared
        ? attentionType === "full_attention"
          ? LAST_FULL_KV_SOURCE_LAYER
          : LAST_SLIDING_KV_SOURCE_LAYER
        : layerIndex,
      q: projection("self_attn.q_proj"),
      k: isKvShared ? null : projection("self_attn.k_proj"),
      v: isKvShared ? null : projection("self_attn.v_proj"),
      output: projection("self_attn.o_proj"),
      qNorm: tensor("self_attn.q_norm.weight"),
      kNorm: layerIndex < K_NORM_LAYER_COUNT ? tensor("self_attn.k_norm.weight") : null,
    },
    mlp: {
      bits: int2Mlp ? 2 : 4,
      intermediateSize: int2Mlp ? 12288 : 6144,
      gate: projection("mlp.gate_proj"),
      up: projection("mlp.up_proj"),
      down: projection("mlp.down_proj"),
    },
    ple: {
      inputGate: projection("per_layer_input_gate"),
      projection: projection("per_layer_projection"),
    },
    norms: {
      input: tensor("input_layernorm.weight"),
      postAttention: tensor("post_attention_layernorm.weight"),
      preFeedforward: tensor("pre_feedforward_layernorm.weight"),
      postFeedforward: tensor("post_feedforward_layernorm.weight"),
      postPerLayerInput: tensor("post_per_layer_input_norm.weight"),
    },
    layerScalar: tensor("layer_scalar"),
    tensorNames: contracts.map(({ name }) => name),
    tensorBytes: contracts.reduce((total, { name }) => total + requiredTensor(tensors, name).byteLength, 0),
  };
}

export function createGemmaLayerPlans(
  descriptors: ReadonlyMap<string, CachedTensorDescriptor>,
): GemmaLayerPlan[] {
  return Array.from({ length: LAYER_COUNT }, (_, layerIndex) =>
    createGemmaLayerPlan(descriptors, layerIndex));
}

function addProjectionContracts(
  contracts: GemmaTensorContract[],
  prefix: string,
  weightDtype: "I8" | "U8",
  weightShape: readonly number[],
  outputFeatures: number,
): void {
  contracts.push(
    contract(`${prefix}.input_activation_scale`, "F32", []),
    contract(`${prefix}.output_activation_scale`, "F32", []),
    contract(`${prefix}.weight_scale`, "F32", [outputFeatures, 1]),
    contract(`${prefix}.weight`, weightDtype, weightShape),
  );
}

function contract(
  name: string,
  dtype: TensorDtype,
  shape: readonly number[],
): GemmaTensorContract {
  return { name, dtype, shape };
}

function validateDescriptor(
  actual: CachedTensorDescriptor,
  expected: GemmaTensorContract,
  layerIndex: number,
): void {
  const expectedBytes = expected.shape.reduce((product, dimension) => product * dimension, 1) *
    bytesPerElement(expected.dtype);
  if (
    actual.name !== expected.name ||
    actual.dtype !== expected.dtype ||
    !sameShape(actual.shape, expected.shape) ||
    actual.byteLength !== expectedBytes ||
    actual.end - actual.begin !== actual.byteLength
  ) {
    throw new Error(
      `Gemma layer ${layerIndex} tensor contract mismatch for ${expected.name}: ` +
      `expected ${expected.dtype}[${expected.shape.join(",")}] ${expectedBytes} bytes, ` +
      `received ${actual.dtype}[${actual.shape.join(",")}] ${actual.byteLength} bytes`,
    );
  }
}

function bytesPerElement(dtype: TensorDtype): number {
  return dtype === "F32" ? 4 : dtype === "BF16" ? 2 : 1;
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length &&
    actual.every((dimension, index) => dimension === expected[index]);
}

function requiredTensor(
  tensors: ReadonlyMap<string, CachedTensorDescriptor>,
  name: string,
): CachedTensorDescriptor {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Validated Gemma tensor ${name} is unavailable`);
  return tensor;
}

function isFullAttentionLayer(layerIndex: number): boolean {
  return layerIndex % 5 === 4;
}

function assertLayerIndex(layerIndex: number): void {
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= LAYER_COUNT) {
    throw new Error(`Gemma layer index must be an integer from 0 through ${LAYER_COUNT - 1}`);
  }
}