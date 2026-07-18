import { expect, test } from "@playwright/test";

test("blocks a broad speedup claim when a required runtime is not measured", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/benchmark/e2b-performance-proof.ts";
    const { canClaimE2BBroadSuperiority, renderE2BPerformanceProofMarkdown } =
      await import(modulePath);
    const metric = { median: 10, p95: 12 };
    const artifact = {
      schemaVersion: 1 as const,
      capturedAt: "2026-07-17T00:00:00.000Z",
      status: "partial" as const,
      model: { id: "google/gemma-4-E2B", revision: "pinned" },
      environment: { userAgent: "Chrome", adapter: "Apple", device: "Mac" },
      methodology: ["Separate runtime pages."],
      runtimes: [
        { id: "owned-webgpu", label: "Owned", version: "local", modelArtifact: "QAT", artifactEquivalence: "pinned-source-equivalent", status: "same-device-measured", notes: [] },
        { id: "pinned-hugging-face-webgpu", label: "HF", version: "commit", modelArtifact: "QAT", artifactEquivalence: "pinned-source-equivalent", status: "same-device-measured", notes: [] },
        { id: "transformers-js", label: "Transformers.js", version: "4.2.0", modelArtifact: "No graph", artifactEquivalence: "unverified", status: "blocked", notes: ["No equivalent export."] },
        { id: "litert-lm-web", label: "LiteRT-LM", version: "0.14.0", modelArtifact: ".litertlm", artifactEquivalence: "model-family-only", status: "published-reference", notes: [] },
      ],
      cases: [{
        id: "case",
        promptTokens: 19,
        expectedOutputTokens: 11,
        results: [
          { runtimeId: "owned-webgpu", evidenceStatus: "same-device-measured", exactOutputMatch: true, generatedTokens: 11, metrics: { ttftMs: metric, itlMs: metric, tpotMs: metric, decodeTokensPerSecond: 100, totalMs: metric } },
          { runtimeId: "pinned-hugging-face-webgpu", evidenceStatus: "same-device-measured", exactOutputMatch: true, generatedTokens: 11, metrics: { ttftMs: { median: 20, p95: 22 }, itlMs: { median: 20, p95: 22 }, tpotMs: { median: 20, p95: 22 }, decodeTokensPerSecond: 50, totalMs: { median: 20, p95: 22 } } },
        ],
      }],
      publishedReferences: [],
    };
    return {
      claim: canClaimE2BBroadSuperiority(artifact),
      markdown: renderE2BPerformanceProofMarkdown(artifact),
    };
  });

  expect(result.claim).toBe(false);
  expect(result.markdown).toContain("No blanket performance-superiority claim is supported");
  expect(result.markdown).toContain("Transformers.js");
  expect(result.markdown).toContain("No equivalent export.");
  expect(result.markdown).toContain("10 / 12 ms");
  expect(result.markdown).toContain("Owned relative delta");
});