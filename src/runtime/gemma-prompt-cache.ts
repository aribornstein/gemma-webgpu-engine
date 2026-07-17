export function reusableGemmaPromptPrefixLength(
  promptTokenIds: readonly number[],
  evaluatedTokenIds: readonly number[],
): number {
  const maximumReusable = Math.max(0, promptTokenIds.length - 1);
  let prefixLength = 0;
  while (prefixLength < maximumReusable &&
      prefixLength < evaluatedTokenIds.length &&
      promptTokenIds[prefixLength] === evaluatedTokenIds[prefixLength]) {
    prefixLength += 1;
  }
  return prefixLength;
}

export function sameGemmaMultimodalIdentity(
  current: readonly string[],
  retained: readonly string[],
): boolean {
  return current.length === retained.length &&
    current.every((identity, index) => identity === retained[index]);
}