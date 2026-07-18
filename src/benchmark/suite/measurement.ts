import type {
  BenchmarkAdapter,
  BenchmarkCase,
  ExternalGenerationMetrics,
  GenerationResult,
  RuntimeMetric,
} from "./types";

export interface MeasuredGeneration {
  result: GenerationResult;
  external: ExternalGenerationMetrics;
  nativeMetrics: readonly RuntimeMetric[];
  chunks: readonly { text: string; timestampMs: number }[];
}

export async function measureGeneration(
  adapter: BenchmarkAdapter,
  testCase: BenchmarkCase,
): Promise<MeasuredGeneration> {
  let requestStartMs: number | null = null;
  const chunks: { text: string; timestampMs: number }[] = [];
  const nativeMetrics: RuntimeMetric[] = [];
  const result = await adapter.generate(testCase, {
    onRequestStart(timestampMs) {
      if (requestStartMs !== null) throw new Error("Adapter reported request start twice");
      requestStartMs = timestampMs;
    },
    onTextChunk(text, timestampMs) {
      if (text.length > 0) chunks.push({ text, timestampMs });
    },
    onRuntimeMetric(metric) {
      nativeMetrics.push(metric);
    },
  });
  const completionMs = performance.now();
  if (requestStartMs === null) throw new Error(`${adapter.id} did not report request start`);
  const retokenizedOutputTokens = await adapter.countTokens(result.text, "output");
  const retokenizedResult = { ...result, outputTokens: retokenizedOutputTokens };
  const firstVisibleOutputMs = chunks[0]?.timestampMs ?? null;
  const chunkIntervals = chunks.slice(1).map((chunk, index) =>
    round(chunk.timestampMs - chunks[index].timestampMs)
  );
  const decodeDurationMs = firstVisibleOutputMs === null
    ? 0
    : completionMs - firstVisibleOutputMs;
  const decodeTokenCount = Math.max(0, retokenizedOutputTokens - (firstVisibleOutputMs === null ? 0 : 1));
  const totalMs = completionMs - requestStartMs;
  return {
    result: retokenizedResult,
    external: {
      requestStartMs: round(requestStartMs),
      firstVisibleOutputMs: firstVisibleOutputMs === null ? null : round(firstVisibleOutputMs),
      completionMs: round(completionMs),
      ttftMs: firstVisibleOutputMs === null ? null : round(firstVisibleOutputMs - requestStartMs),
      totalMs: round(totalMs),
      aggregateDecodeTokensPerSecond: decodeDurationMs > 0
        ? round(decodeTokenCount * 1000 / decodeDurationMs)
        : null,
      charactersPerSecond: totalMs > 0 ? round(retokenizedResult.text.length * 1000 / totalMs) : 0,
      streamChunkCount: chunks.length,
      streamChunkIntervalMs: Object.freeze(chunkIntervals),
    },
    nativeMetrics: Object.freeze(nativeMetrics),
    chunks: Object.freeze(chunks),
  };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}