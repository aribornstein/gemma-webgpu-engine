import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { GemmaGenerationSession } from "../src/runtime/gemma-session";
import type { GemmaResearchStage } from "../src/research/engine-types";

test("within-layer captures and patches are active, aligned, and reversible on the real model", async ({ page }) => {
  test.skip(process.env.COMPONENT_LIVE !== "1" || !existsSync(resolve("public/models/gemma-4-e2b/model.safetensors")), "Set COMPONENT_LIVE=1 with the local checkpoint");
  test.setTimeout(300_000);
  await page.goto("/research.html");
  const result = await page.evaluate(async () => {
    const path = "/src/runtime/gemma-session.ts";
    const { loadGemmaGenerationSession } = await import(path);
    const session: GemmaGenerationSession = await loadGemmaGenerationSession({
      sourceUrl: "/models/gemma-4-e2b/model.safetensors", cacheCapacity: 512, prefillStrategy: "sequential",
    });
    try {
      const tokens = session.researchTokenize([{ role: "user", content: "Name the three primary colors in one short sentence." }]);
      const prompt = tokens.promptTokenIds;
      const positions = [prompt.length - 1];
      const stages: GemmaResearchStage[] = ["layer-input", "attention", "mlp", "layer-output"];
      const base = await session.researchEvaluate({ promptTokenIds: prompt, generate: { maxNewTokens: 16 } });
      const captured = await session.researchEvaluate({
        promptTokenIds: prompt, generate: { maxNewTokens: 16 }, capture: { layers: [20, 34], positions, stages },
      });
      const checks = [];
      for (const stage of ["attention", "mlp", "layer-output"] as const) {
        const values = captured.captures.find((c) => typeof c.site === "object" && c.site.residual === 34 && c.stage === stage)!.values;
        const patch = { layer: 34, stage, vector: values, mode: "replace" as const, amount: 1, tokenPosition: prompt.length - 1 };
        const sham = await session.researchEvaluate({ promptTokenIds: prompt, generate: { maxNewTokens: 16 }, patch });
        const active = await session.researchEvaluate({
          promptTokenIds: prompt, generate: { maxNewTokens: 1 },
          patch: { ...patch, vector: values.map((v) => -v) },
        });
        checks.push({
          stage, shamExact: JSON.stringify(sham.tokenIds) === JSON.stringify(base.tokenIds),
          active: JSON.stringify(active.firstDecision) !== JSON.stringify(base.firstDecision),
        });
      }
      const after = await session.researchEvaluate({ promptTokenIds: prompt, generate: { maxNewTokens: 16 } });
      return {
        checks, captures: captured.captures.length,
        captureParity: JSON.stringify(captured.tokenIds) === JSON.stringify(base.tokenIds),
        revert: JSON.stringify(after.tokenIds) === JSON.stringify(base.tokenIds),
        finite: captured.captures.every((c) => c.values.length === 1536 && c.values.every(Number.isFinite)),
      };
    } finally { session.destroy(); }
  });
  expect(result.captures).toBe(8);
  expect(result.captureParity && result.revert && result.finite).toBe(true);
  for (const check of result.checks) expect(check, check.stage).toMatchObject({ shamExact: true, active: true });
});

test("real-model paired replay matches a zero overlay and leaves the baseline unchanged", async ({ page }) => {
  test.skip(process.env.COMPONENT_LIVE !== "1" || !existsSync(resolve("public/models/gemma-4-e2b/model.safetensors")), "Set COMPONENT_LIVE=1 with the local checkpoint");
  test.setTimeout(300_000);
  await page.goto("/research.html");
  const result = await page.evaluate(async () => {
    await window.researchWorkbench.load();
    try {
      const request = {
        id: "live-zero-pair", op: "pairedReplay" as const, protocol: "fixed-prefix" as const,
        messages: [{ role: "system" as const, content: "Answer in one short sentence." }],
        playerTurns: ["Name three colors."], maxNewTokens: 12, temperature: 0, seed: 23,
        arms: [
          { id: "base", label: "Baseline" },
          { id: "zero", label: "Zero overlay", overlays: [{ layer: 20, a: Array(1536).fill(0), b: Array(1536).fill(0), coefficient: 0.1 }] },
        ],
      };
      const first = await window.researchWorkbench.run(request);
      const patch = await (await fetch("/research/agent_p6.json")).json() as {
        overlays: { site: { layer: number }; a: number[]; b: number[]; scale: number }[];
      };
      const active = await window.researchWorkbench.run({
        ...request, id: "live-historical-pair",
        arms: [request.arms[0], { id: "p6", label: "Historical unpromoted p6", overlays: patch.overlays.map((o) => ({
          layer: o.site.layer, a: o.a, b: o.b, coefficient: o.scale,
        })) }],
      });
      const after = await window.researchWorkbench.run({ ...request, id: "live-zero-pair-revert" });
      return { first, active, after };
    } finally { await window.researchWorkbench.close(); }
  });
  const first = result.first.result as { complete: boolean; arms: { turns: { tokenIds: number[]; tokenLogProbs: number[]; promptHash: string }[] }[] };
  const after = result.after.result as typeof first;
  expect(result.first.status).toBe("completed");
  expect(first.complete).toBe(true);
  expect(first.arms[0].turns[0].tokenIds.length).toBeGreaterThan(0);
  expect(first.arms[0].turns[0].tokenIds).toEqual(first.arms[1].turns[0].tokenIds);
  expect(first.arms[0].turns[0].tokenLogProbs).toEqual(first.arms[1].turns[0].tokenLogProbs);
  expect(first.arms[0].turns[0].promptHash).toBe(first.arms[1].turns[0].promptHash);
  const active = result.active.result as typeof first;
  expect(result.active.status).toBe("completed");
  expect(active.arms[0].turns[0].tokenIds).toEqual(first.arms[0].turns[0].tokenIds);
  expect(active.arms[1].turns[0].tokenLogProbs).not.toEqual(active.arms[0].turns[0].tokenLogProbs);
  expect(after.arms[0].turns[0].tokenIds).toEqual(first.arms[0].turns[0].tokenIds);
});
