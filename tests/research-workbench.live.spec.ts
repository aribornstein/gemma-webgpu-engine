import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

test("live workbench has exact sham/revert parity, bounded weights and lifecycle recovery", async ({ page }) => {
  test.skip(process.env.MECHANISTIC_LIVE !== "1" || !existsSync(resolve("public/models/gemma-4-e2b/model.safetensors")), "Requires MECHANISTIC_LIVE=1 and pinned local checkpoint");
  test.setTimeout(300_000);
  await page.goto("/research.html");
  const checks = await page.evaluate(async () => {
    const api = window.researchWorkbench;
    await api.load({
      runId: `live-${Date.now()}`, revision: "live-test",
      checkpointRevision: "9fcec64df66cb1e4d972fc5cdc142afb25b2362c",
      checkpointSha256: "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4",
      environment: navigator.userAgent,
    });
    const messages = [{ role: "user" as const, content: "Name the three primary colors in one short sentence." }];
    const base = await api.run({ id: "base", op: "replay", messages, maxNewTokens: 16 });
    const capture = await api.run({ id: "capture", op: "capture", messages, layers: [20, 34] });
    const data = capture.result as { captures: { values: number[] }[] };
    const patch = {
      op: "patch" as const, messages, layer: 20, mode: "replace" as const,
      vector: data.captures[0].values, sourceArtifactIds: [capture.id], maxNewTokens: 16,
    };
    const sham = await api.run({ id: "sham", ...patch, amount: 1 });
    const zero = await api.run({ id: "zero", ...patch, amount: 0 });
    const active = await api.run({
      id: "active-sentinel", ...patch, layer: 34, amount: 1,
      vector: data.captures[1].values.map((v) => -v),
    });
    const reverted = await api.run({ id: "reverted", op: "replay", messages, maxNewTokens: 16 });
    const comparable = (result: unknown) => JSON.stringify((result as { tokenIds: number[] }).tokenIds);
    const scored = await api.run({ id: "score", op: "score", messages, continuation: "The primary colors are red, yellow, and blue." });
    const weights = await api.run({
      id: "weights", op: "inspectWeights",
      slice: { layer: 20, projection: "down", rowStart: 2, rowCount: 2, columnStart: 0, columnCount: 32 },
    });
    const weightData = weights.result as { values: number[]; codes: number[]; packedBits: number; bytesRead: number };
    const counts = (scored.result as { tokenLogProbs: number[] }).tokenLogProbs;
    await api.run({ id: "compare", op: "compare", artifactIds: [base.id, reverted.id] });
    const cancelRequest = api.run({
      id: "cancelled", op: "replay",
      messages: [{ role: "user", content: "Write a long story about a dragon." }], maxNewTokens: 128,
    });
    const cancellation = cancelRequest.then(() => "unexpected completion", (error: Error) => error.name);
    setTimeout(() => api.cancel(), 100);
    const cancelled = await cancellation;
    const recovered = await api.run({ id: "recovered", op: "replay", messages, maxNewTokens: 16 });
    const output = {
      shamExact: comparable(sham.result) === comparable(base.result),
      zeroExact: comparable(zero.result) === comparable(base.result),
      nonzeroActuallyActive: JSON.stringify((active.result as { firstDecision: unknown }).firstDecision) !==
        JSON.stringify((base.result as { firstDecision: unknown }).firstDecision),
      revertExact: comparable(reverted.result) === comparable(base.result),
      recoveredExact: comparable(recovered.result) === comparable(base.result),
      captureValid: data.captures.length === 2 && data.captures.every((c) => c.values.length === 1536 && c.values.every(Number.isFinite) && c.values.some((v) => v !== 0)),
      scoreFinite: counts.length > 1 && counts.every(Number.isFinite),
      weightsValid: weightData.packedBits === 2 && weightData.values.length === 64 && weightData.codes.every((v) => v >= -2 && v <= 1) && weightData.bytesRead < 1024,
      cancelled,
    };
    await api.close();
    return output;
  });
  expect(checks).toEqual({
    shamExact: true, zeroExact: true, nonzeroActuallyActive: true, revertExact: true, recoveredExact: true,
    captureValid: true, scoreFinite: true, weightsValid: true, cancelled: "AbortError",
  });
});
