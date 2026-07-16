import type {
  CachedTensorPayload,
  CachedTensorSliceRequest,
} from "./cached-safetensors";
import { bfloat16ToFloat32 } from "./gemma-layer-materializer";
import type { GemmaLayerTensorSource } from "./gemma-layer-weights";

const HIDDEN_SIZE = 1536;
const LAYER_COUNT = 35;
const PER_LAYER_SIZE = 256;
const PER_LAYER_TOTAL = LAYER_COUNT * PER_LAYER_SIZE;
const TOKEN_BITS = "model.language_model.embed_tokens.embedding_quantized";
const TOKEN_SCALES = "model.language_model.embed_tokens.embedding_scale";
const LAYER_BITS = "model.language_model.embed_tokens_per_layer.embedding_quantized";
const LAYER_SCALES = "model.language_model.embed_tokens_per_layer.embedding_scale";
const PROJECTION = "model.language_model.per_layer_model_projection.weight";
const PROJECTION_NORM = "model.language_model.per_layer_projection_norm.weight";

export interface GemmaInputTensorSource extends GemmaLayerTensorSource {
  readTensorSlice(name: string, byteOffset: number, byteLength: number): Promise<Uint8Array>;
  readTensorSlices?(requests: readonly CachedTensorSliceRequest[]): Promise<Uint8Array[]>;
}

export interface GemmaInputWeights {
  projectionBfloat16: Uint32Array;
  projectionNorm: Float32Array;
}

export interface GemmaTokenInputs {
  hidden: Float32Array;
  perLayerEmbedding: Float32Array;
}

export async function loadGemmaInputWeights(
  source: GemmaInputTensorSource,
): Promise<GemmaInputWeights> {
  validateDescriptor(source, PROJECTION, "BF16", [PER_LAYER_TOTAL, HIDDEN_SIZE]);
  validateDescriptor(source, PROJECTION_NORM, "BF16", [PER_LAYER_SIZE]);
  const tensors = await source.readTensors([PROJECTION, PROJECTION_NORM]);
  const projection = required(tensors, PROJECTION);
  const norm = required(tensors, PROJECTION_NORM);
  return {
    projectionBfloat16: packedUint32(projection.bytes),
    projectionNorm: bfloat16ToFloat32(norm),
  };
}

export async function loadGemmaTokenInputs(
  source: GemmaInputTensorSource,
  tokenId: number,
): Promise<GemmaTokenInputs> {
  return (await loadGemmaTokenInputBatch(source, [tokenId], 1))[0];
}

export async function loadGemmaTokenInputBatch(
  source: GemmaInputTensorSource,
  tokenIds: readonly number[],
  rowCount = tokenIds.length,
): Promise<GemmaTokenInputs[]> {
  if (!Number.isInteger(rowCount) || rowCount < 1 || tokenIds.length < 1 ||
      tokenIds.length > rowCount) {
    throw new Error("Gemma token input batch requires tokens within a positive row count");
  }
  for (const tokenId of tokenIds) validateTokenId(tokenId);
  validateDescriptor(source, TOKEN_BITS, "U8", [262144, HIDDEN_SIZE / 4]);
  validateDescriptor(source, TOKEN_SCALES, "F32", [262144, 1]);
  validateDescriptor(source, LAYER_BITS, "U8", [262144, PER_LAYER_TOTAL / 2]);
  validateDescriptor(source, LAYER_SCALES, "F32", [262144, LAYER_COUNT]);
  const paddedTokenIds = Array.from({ length: rowCount }, (_, row) => tokenIds[row] ?? 0);
  const requests = paddedTokenIds.flatMap((tokenId): CachedTensorSliceRequest[] => [
    { name: TOKEN_BITS, byteOffset: tokenId * (HIDDEN_SIZE / 4), byteLength: HIDDEN_SIZE / 4 },
    { name: TOKEN_SCALES, byteOffset: tokenId * 4, byteLength: 4 },
    {
      name: LAYER_BITS,
      byteOffset: tokenId * (PER_LAYER_TOTAL / 2),
      byteLength: PER_LAYER_TOTAL / 2,
    },
    {
      name: LAYER_SCALES,
      byteOffset: tokenId * LAYER_COUNT * 4,
      byteLength: LAYER_COUNT * 4,
    },
  ]);
  const slices = source.readTensorSlices
    ? await source.readTensorSlices(requests)
    : await Promise.all(requests.map(({ name, byteOffset, byteLength }) =>
        source.readTensorSlice(name, byteOffset, byteLength)));
  return paddedTokenIds.map((_, row) => dequantizeTokenInputs(
    slices[row * 4],
    slices[row * 4 + 1],
    slices[row * 4 + 2],
    slices[row * 4 + 3],
  ));
}

function dequantizeTokenInputs(
  tokenBits: Uint8Array,
  tokenScaleBytes: Uint8Array,
  layerBits: Uint8Array,
  layerScaleBytes: Uint8Array,
): GemmaTokenInputs {
  const tokenScale = float32Scalar(tokenScaleBytes);
  const layerScales = float32Array(layerScaleBytes);
  const hidden = new Float32Array(HIDDEN_SIZE);
  const embedScale = Math.fround(Math.sqrt(HIDDEN_SIZE));
  for (let index = 0; index < HIDDEN_SIZE; index += 1) {
    const code = (tokenBits[index >> 2] >> ((index & 3) * 2)) & 3;
    hidden[index] = Math.fround(
      Math.fround(embedScale * tokenScale) * Math.fround(code - 2),
    );
  }
  const perLayerEmbedding = new Float32Array(PER_LAYER_TOTAL);
  const perLayerScale = Math.fround(Math.sqrt(PER_LAYER_SIZE));
  for (let layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex += 1) {
    const scale = Math.fround(perLayerScale * layerScales[layerIndex]);
    const layerOffset = layerIndex * PER_LAYER_SIZE;
    for (let element = 0; element < PER_LAYER_SIZE; element += 1) {
      const index = layerOffset + element;
      const code = (layerBits[index >> 1] >> ((index & 1) * 4)) & 15;
      perLayerEmbedding[index] = Math.fround(scale * Math.fround(code - 8));
    }
  }
  return { hidden, perLayerEmbedding };
}

function validateTokenId(tokenId: number): void {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= 262144) {
    throw new Error("Gemma token ID must be an integer below 262144");
  }
}

function required(
  tensors: ReadonlyMap<string, CachedTensorPayload>,
  name: string,
): CachedTensorPayload {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Gemma input load omitted tensor ${name}`);
  return tensor;
}

function validateDescriptor(
  source: GemmaLayerTensorSource,
  name: string,
  dtype: string,
  shape: readonly number[],
): void {
  const descriptor = source.descriptors.get(name);
  if (!descriptor || descriptor.dtype !== dtype ||
      descriptor.shape.length !== shape.length ||
      descriptor.shape.some((value, index) => value !== shape[index])) {
    throw new Error(`Gemma input tensor contract mismatch for ${name}`);
  }
}

function packedUint32(bytes: Uint8Array): Uint32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("BF16 projection storage is not u32 aligned");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint32Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  );
}

function float32Scalar(bytes: Uint8Array): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(0, true);
}

function float32Array(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}