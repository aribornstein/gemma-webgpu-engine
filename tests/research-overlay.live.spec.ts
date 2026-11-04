import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

// Research adapter gate (G1): identity/revert parity of both overlay sites, teacher-forced scoring
// against generation log-probabilities, capture alignment, and head-projection linearity.
// Requires the checkpoint at public/models/gemma-4-e2b/model.safetensors.
const LIVE = process.env.RESEARCH_LIVE === "1";
const LOCAL_CHECKPOINT = resolve("public/models/gemma-4-e2b/model.safetensors");

test("research overlays and probes preserve inference parity", async ({ page }) => {
  test.skip(!LIVE || !existsSync(LOCAL_CHECKPOINT), "Set RESEARCH_LIVE=1 and provide the local checkpoint");
  test.setTimeout(600_000);
  await page.goto("/");
  const report = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-session.ts";
      const { loadGemmaGenerationSession } = await import(modulePath);
    const s = await loadGemmaGenerationSession({ sourceUrl: "/models/gemma-4-e2b/model.safetensors", cacheCapacity: 2048, prefillStrategy: "sequential" });
    const out: Record<string, any> = {}; const H = 1536;
    const rand = (seed: number) => { let x = seed; return () => ((x = (x * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5; };
    const vec = (seed: number) => { const r = rand(seed); const v = new Float32Array(H); for (let i = 0; i < H; i++) v[i] = r(); const n = Math.hypot(...v); return v.map((x) => x / n); };
    const prompt = "Name the three primary colors in one short sentence.";
    const gen = async (o = {}) => (await s.generate(prompt, { maxNewTokens: 16, temperature: 0, reusePromptCache: false, ...o })).generatedTokenIds.join(",");
    console.log("RESEARCH 10 baseline");
    const base = await gen();
    out.baseGreedy = base;
    out.baseLogits = await gen({ captureTokenLogProbabilities: true });
    out.identityResidual = await (async () => { await s.setResearchOverlay({ site: { layer: 22 }, a: vec(1), b: vec(2), coefficient: 0 }); return gen(); })();
    out.identityFinal = await (async () => { await s.setResearchOverlay({ site: "final", a: vec(1), b: vec(2), coefficient: 0 }); return gen(); })();
    console.log("RESEARCH 30 active overlays");
    await s.setResearchOverlay({ site: { layer: 22 }, a: vec(3), b: vec(4), coefficient: 400 });
    out.activeResidual = await gen();
    await s.setResearchOverlay(null);
    out.revertResidual = await gen();
    await s.setResearchOverlay({ site: "final", a: vec(3), b: vec(4), coefficient: 400 });
    out.activeFinal = await gen();
    await s.setResearchOverlay(null);
    out.revertFinal = await gen();
    console.log("RESEARCH 50 teacher forcing");
    const r = await s.generate(prompt, { maxNewTokens: 12, temperature: 0, reusePromptCache: false, captureTokenLogProbabilities: true });
    const sc = await s.researchScoreContinuation(r.promptTokenIds, r.generatedTokenIds);
    out.tfMaxAbsDiff = Math.max(...sc.tokenLogProbs.map((v: number, i: number) => Math.abs(v - r.tokenLogProbabilities[i])));
    out.tfCount = sc.tokenCount;
    console.log("RESEARCH 70 capture alignment");
    const ids = r.promptTokenIds; const P = ids.length - 4;
    const c1 = await s.researchCaptureActivations(ids, [P, ids.length - 1], [12, 22, "final"]);
    const c2 = await s.researchCaptureActivations(ids.slice(0, P + 1), [P], [12, 22, "final"]);
    const c3 = await s.researchCaptureActivations(ids, [P, ids.length - 1], [12, 22, "final"]);
    const same = (x: { values: Float32Array }, y: { values: Float32Array }) => x.values.length === y.values.length && x.values.every((v: number, i: number) => Object.is(v, y.values[i]));
    out.captureCount = c1.length;
    out.capturePrefixEqual = c2.every((c: { values: Float32Array }, i: number) => same(c, c1[i]));
    out.captureRepeatEqual = c1.every((c: { values: Float32Array }, i: number) => same(c, c3[i]));
    out.captureFiniteNonzero = c1.every((c: { values: Float32Array }) => c.values.every(Number.isFinite) && c.values.some((v) => v !== 0));
    const r34 = await s.researchCaptureActivations(ids, [P], [{ residual: 34 }, "residual"]);
  out.residualCaptureMatches = same(r34[0], r34[1]);
  console.log("RESEARCH 85 head projection linearity");
    const b = vec(5); const u1 = await s.researchHeadProjection(b); const u2 = await s.researchHeadProjection(b.map((x: number) => 2 * x));
    let maxRel = 0; for (let i = 0; i < u1.length; i += 997) maxRel = Math.max(maxRel, Math.abs(u2[i] - 2 * u1[i]) / (Math.abs(u2[i]) + 1e-6));
    out.headLinearMaxRel = maxRel;
    // final overlay CPU reference: logits_overlay - logits_base == c * (a.h) * u
    const a = vec(6); const coeff = 3;
    const base2 = await s.researchScoreContinuation(r.promptTokenIds, [r.generatedTokenIds[0]]);
    await s.setResearchOverlay({ site: "final", a, b, coefficient: coeff });
    const over2 = await s.researchScoreContinuation(r.promptTokenIds, [r.generatedTokenIds[0]]);
    await s.setResearchOverlay(null);
    out.finalOverlayChangesLogProb = base2.tokenLogProbs[0] !== over2.tokenLogProbs[0];
    out.afterAllGreedy = await gen();
      s.destroy();
    return out;
  });
  expect(report.baseLogits).toBe(report.baseGreedy);
  expect(report.identityResidual).toBe(report.baseGreedy);
  expect(report.identityFinal).toBe(report.baseGreedy);
  expect(report.revertResidual).toBe(report.baseGreedy);
  expect(report.revertFinal).toBe(report.baseGreedy);
  expect(report.activeResidual).not.toBe(report.baseGreedy);
  expect(report.activeFinal).not.toBe(report.baseGreedy);
  expect(report.tfMaxAbsDiff).toBeLessThan(1e-9);
  expect(report.capturePrefixEqual && report.captureRepeatEqual && report.captureFiniteNonzero).toBe(true);
  expect(report.headLinearMaxRel).toBeLessThan(1e-4);
  expect(report.residualCaptureMatches).toBe(true);
  expect(report.finalOverlayChangesLogProb).toBe(true);
  expect(report.afterAllGreedy).toBe(report.baseGreedy);
});
