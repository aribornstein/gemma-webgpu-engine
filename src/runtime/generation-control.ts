export interface GemmaGenerationUpdate {
  tokenId: number;
  tokenIndex: number;
  generatedTokenIds: readonly number[];
  text: string;
}

export type GemmaGenerationTokenHandler = (
  update: GemmaGenerationUpdate,
) => void | Promise<void>;

export function throwIfGemmaGenerationAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Gemma generation aborted", "AbortError");
}

export async function emitGemmaGenerationUpdate(
  tokenId: number,
  generatedTokenIds: readonly number[],
  decodeTokens: (tokenIds: readonly number[]) => string,
  onToken?: GemmaGenerationTokenHandler,
): Promise<void> {
  if (!onToken) return;
  const tokenIds = Object.freeze([...generatedTokenIds]);
  await onToken({
    tokenId,
    tokenIndex: tokenIds.length - 1,
    generatedTokenIds: tokenIds,
    text: decodeTokens(tokenIds),
  });
}