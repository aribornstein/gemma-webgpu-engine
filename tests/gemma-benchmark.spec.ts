import { expect, test } from "@playwright/test";

test("summarizes retained full-generation timing samples", async ({ page }) => {
  await page.goto("/");
  const summary = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-benchmark.ts";
    const { summarizeGemmaBenchmarkSamples } = await import(modulePath);
    const timing = (
      prefillMs: number,
      timeToFirstTokenMs: number,
      decodeTokenMs: number[],
      totalMs: number,
    ) => ({
      requestSetupMs: 1,
      cacheResetMs: 2,
      prefillMs,
      prefillMode: "fixed-32" as const,
      timeToFirstTokenMs,
      decodeTokenMs,
      interTokenLatencyMs: decodeTokenMs,
      timePerOutputTokenMs: decodeTokenMs.reduce((sum, value) => sum + value, 0) /
        decodeTokenMs.length,
      logitsReadbackMs: 0,
      callbackMs: 0,
      totalMs,
    });
    return summarizeGemmaBenchmarkSamples([
      {
        iteration: 0,
        timing: timing(100, 120, [200, 300], 1000),
        generatedTokenIds: [10, 11],
        text: "one",
        stopReason: "end-token",
        exactGoldenMatch: true,
      },
      {
        iteration: 1,
        timing: timing(80, 100, [100, 200], 800),
        generatedTokenIds: [10, 11],
        text: "one",
        stopReason: "end-token",
        exactGoldenMatch: true,
      },
    ]);
  });

  expect(summary).toEqual({
    prefill: { medianMs: 80, p95Ms: 100, averageMs: 90 },
    timeToFirstToken: { medianMs: 100, p95Ms: 120, averageMs: 110 },
    decodeToken: { medianMs: 200, p95Ms: 300, averageMs: 200 },
    interTokenLatency: { medianMs: 200, p95Ms: 300, averageMs: 200 },
    timePerOutputToken: { medianMs: 150, p95Ms: 250, averageMs: 200 },
    total: { medianMs: 800, p95Ms: 1000, averageMs: 900 },
    warmDecodeTokensPerSecond: 5,
    endToEndTokensPerSecond: 2.222,
  });
});

test("calculates single-run generation throughput", async ({ page }) => {
  await page.goto("/");
  const throughput = await page.evaluate(async () => {
    const modulePath = "/src/runtime/generation-throughput.ts";
    const { calculateGemmaGenerationThroughput } = await import(modulePath);
    return calculateGemmaGenerationThroughput([
      { decodeTokenMs: [200, 300], totalMs: 1200 },
    ], 3);
  });

  expect(throughput).toEqual({
    warmDecodeTokensPerSecond: 4,
    endToEndTokensPerSecond: 2.5,
  });
});
