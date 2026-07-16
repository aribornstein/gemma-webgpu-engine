const ATTENTION_CHUNK_COUNT = 32;

export function gemmaDecodeAttentionChunkCount(position: number, window: number): number {
  if (!Number.isInteger(position) || position < 0 ||
      !Number.isInteger(window) || window < 0) {
    throw new Error("Decode attention chunk geometry is invalid");
  }
  const activeKeys = window > 0 ? Math.min(position + 1, window) : position + 1;
  return Math.min(ATTENTION_CHUNK_COUNT, Math.max(8, Math.ceil(activeKeys / 64)));
}