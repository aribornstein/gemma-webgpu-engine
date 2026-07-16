import type { GemmaGenerationTiming } from "./gemma-session";

export interface GemmaGenerationThroughput {
  warmDecodeTokensPerSecond: number | null;
  endToEndTokensPerSecond: number | null;
}

export function calculateGemmaGenerationThroughput(
  timings: readonly Pick<GemmaGenerationTiming, "decodeTokenMs" | "totalMs">[],
  generatedTokenCount: number,
): GemmaGenerationThroughput {
  const decodeTokenMs = timings.flatMap((timing) => timing.decodeTokenMs);
  const totalDecodeMs = decodeTokenMs.reduce((sum, duration) => sum + duration, 0);
  const totalGenerationMs = timings.reduce((sum, timing) => sum + timing.totalMs, 0);
  return {
    warmDecodeTokensPerSecond: decodeTokenMs.length > 0 && totalDecodeMs > 0
      ? decodeTokenMs.length * 1000 / totalDecodeMs
      : null,
    endToEndTokensPerSecond: generatedTokenCount > 0 && totalGenerationMs > 0
      ? generatedTokenCount * 1000 / totalGenerationMs
      : null,
  };
}