import type { CachedTensorPayload } from "./cached-safetensors";
import {
  bfloat16ToFloat32,
  float32LittleEndian,
  packedUint32,
} from "./gemma-layer-materializer";
import { PinnedSafetensorsSource } from "./pinned-safetensors";

export const GEMMA_VISION_PATCH_WEIGHT =
  "model.vision_tower.patch_embedder.input_proj.weight";
export const GEMMA_VISION_POSITION_WEIGHT =
  "model.vision_tower.patch_embedder.position_embedding_table";
export const GEMMA_VISION_PROJECTOR_WEIGHT =
  "model.embed_vision.embedding_projection.weight";

export interface GemmaVisionPatchWeights {
  projection: Float32Array;
  positions: Float32Array;
  sourceBytes: number;
}

export interface GemmaVisionProjectionWeights {
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  inputScale: number;
  outputScale: number;
  sourceBytes: number;
}

export interface GemmaVisionLayerNormWeights {
  input: Float32Array;
  postAttention: Float32Array;
  preFeedforward: Float32Array;
  postFeedforward: Float32Array;
  query: Float32Array;
  key: Float32Array;
  sourceBytes: number;
}

export interface GemmaVisionLayerWeights {
  layerIndex: number;
  norms: GemmaVisionLayerNormWeights;
  query: GemmaVisionProjectionWeights;
  key: GemmaVisionProjectionWeights;
  value: GemmaVisionProjectionWeights;
  attentionOutput: GemmaVisionProjectionWeights;
  gate: GemmaVisionProjectionWeights;
  up: GemmaVisionProjectionWeights;
  down: GemmaVisionProjectionWeights;
  sourceBytes: number;
}

export interface GemmaVisionProjectorWeights {
  projection: Float32Array;
  sourceBytes: number;
}

export async function loadGemmaVisionPatchWeights(
  source: PinnedSafetensorsSource,
): Promise<GemmaVisionPatchWeights> {
  validateDescriptor(source, GEMMA_VISION_PATCH_WEIGHT, "BF16", [768, 768]);
  validateDescriptor(source, GEMMA_VISION_POSITION_WEIGHT, "BF16", [2, 10_240, 768]);
  const tensors = await source.readTensors([
    GEMMA_VISION_PATCH_WEIGHT,
    GEMMA_VISION_POSITION_WEIGHT,
  ]);
  const projection = required(tensors, GEMMA_VISION_PATCH_WEIGHT);
  const positions = required(tensors, GEMMA_VISION_POSITION_WEIGHT);
  return {
    projection: bfloat16ToFloat32(projection),
    positions: bfloat16ToFloat32(positions),
    sourceBytes: projection.byteLength + positions.byteLength,
  };
}

export async function loadGemmaVisionProjectionWeights(
  source: PinnedSafetensorsSource,
  prefix: string,
  outFeatures: number,
  inFeatures: number,
): Promise<GemmaVisionProjectionWeights> {
  const weightName = `${prefix}.linear.weight`;
  const scaleName = `${prefix}.linear.weight_scale`;
  const inputScaleName = `${prefix}.linear.input_activation_scale`;
  const outputScaleName = `${prefix}.linear.output_activation_scale`;
  validateDescriptor(source, weightName, "I8", [outFeatures, inFeatures]);
  validateDescriptor(source, scaleName, "F32", [outFeatures, 1]);
  validateDescriptor(source, inputScaleName, "F32", []);
  validateDescriptor(source, outputScaleName, "F32", []);
  const tensors = await source.readTensors([
    weightName,
    scaleName,
    inputScaleName,
    outputScaleName,
  ]);
  const weight = required(tensors, weightName);
  const scale = required(tensors, scaleName);
  const inputScale = float32LittleEndian(required(tensors, inputScaleName));
  const outputScale = float32LittleEndian(required(tensors, outputScaleName));
  if (inputScale.length !== 1 || outputScale.length !== 1) {
    throw new Error(`Gemma vision projection ${prefix} has invalid activation scales`);
  }
  return {
    packedWeights: packedUint32(weight),
    rowScales: float32LittleEndian(scale),
    inputScale: inputScale[0],
    outputScale: outputScale[0],
    sourceBytes: weight.byteLength + scale.byteLength +
      inputScale.byteLength + outputScale.byteLength,
  };
}

export async function loadGemmaVisionLayerNormWeights(
  source: PinnedSafetensorsSource,
  layerIndex: number,
): Promise<GemmaVisionLayerNormWeights> {
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= 16) {
    throw new Error("Gemma vision layer index is invalid");
  }
  const prefix = `model.vision_tower.encoder.layers.${layerIndex}`;
  const names = {
    input: `${prefix}.input_layernorm.weight`,
    postAttention: `${prefix}.post_attention_layernorm.weight`,
    preFeedforward: `${prefix}.pre_feedforward_layernorm.weight`,
    postFeedforward: `${prefix}.post_feedforward_layernorm.weight`,
    query: `${prefix}.self_attn.q_norm.weight`,
    key: `${prefix}.self_attn.k_norm.weight`,
  } as const;
  for (const name of [
    names.input,
    names.postAttention,
    names.preFeedforward,
    names.postFeedforward,
  ]) {
    validateDescriptor(source, name, "BF16", [768]);
  }
  validateDescriptor(source, names.query, "BF16", [64]);
  validateDescriptor(source, names.key, "BF16", [64]);
  const tensors = await source.readTensors(Object.values(names));
  const load = (name: string) => bfloat16ToFloat32(required(tensors, name));
  return {
    input: load(names.input),
    postAttention: load(names.postAttention),
    preFeedforward: load(names.preFeedforward),
    postFeedforward: load(names.postFeedforward),
    query: load(names.query),
    key: load(names.key),
    sourceBytes: Array.from(tensors.values()).reduce(
      (total, tensor) => total + tensor.byteLength,
      0,
    ),
  };
}

export async function loadGemmaVisionLayerWeights(
  source: PinnedSafetensorsSource,
  layerIndex: number,
): Promise<GemmaVisionLayerWeights> {
  const prefix = `model.vision_tower.encoder.layers.${layerIndex}`;
  const [norms, query, key, value, attentionOutput, gate, up, down] =
    await Promise.all([
      loadGemmaVisionLayerNormWeights(source, layerIndex),
      loadGemmaVisionProjectionWeights(source, `${prefix}.self_attn.q_proj`, 768, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.self_attn.k_proj`, 768, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.self_attn.v_proj`, 768, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.self_attn.o_proj`, 768, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.mlp.gate_proj`, 3072, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.mlp.up_proj`, 3072, 768),
      loadGemmaVisionProjectionWeights(source, `${prefix}.mlp.down_proj`, 768, 3072),
    ]);
  const projections = [query, key, value, attentionOutput, gate, up, down];
  return {
    layerIndex,
    norms,
    query,
    key,
    value,
    attentionOutput,
    gate,
    up,
    down,
    sourceBytes: norms.sourceBytes + projections.reduce(
      (total, projection) => total + projection.sourceBytes,
      0,
    ),
  };
}

export async function loadGemmaVisionProjectorWeights(
  source: PinnedSafetensorsSource,
): Promise<GemmaVisionProjectorWeights> {
  validateDescriptor(source, GEMMA_VISION_PROJECTOR_WEIGHT, "F32", [1536, 768]);
  const tensors = await source.readTensors([GEMMA_VISION_PROJECTOR_WEIGHT]);
  const tensor = required(tensors, GEMMA_VISION_PROJECTOR_WEIGHT);
  return {
    projection: float32LittleEndian(tensor),
    sourceBytes: tensor.byteLength,
  };
}

function validateDescriptor(
  source: PinnedSafetensorsSource,
  name: string,
  dtype: string,
  shape: readonly number[],
): void {
  const descriptor = source.descriptor(name);
  if (descriptor.dtype !== dtype || descriptor.shape.join(",") !== shape.join(",")) {
    throw new Error(`Gemma vision tensor ${name} does not match its pinned contract`);
  }
}

function required(
  tensors: ReadonlyMap<string, CachedTensorPayload>,
  name: string,
): CachedTensorPayload {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Gemma vision tensor ${name} was not loaded`);
  return tensor;
}