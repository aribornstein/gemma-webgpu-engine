import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { hasNumericAmount } from "../src/research/workbench/contracts";
import { ResearchWorkbench, type Artifact, type ArtifactStore, type PairedReplayRequest } from "../src/research/workbench/runner";
import type { GemmaResearchEvaluationRequest, GemmaResearchEvaluationResult } from "../src/research/engine-types";

test("recorded comparison studio exposes real pairs, controls, original scales and failure caveats", async ({ page }) => {
  await page.goto("/research.html");
  await expect(page.locator("#comparison-status")).toContainText("26 recorded comparisons");
  await expect(page.locator("#comparison-root")).toContainText("Rejected for language regressions");
  await expect(page.locator("#comparison-root .comparison-card")).toHaveCount(2);
  await expect(page.locator("#comparison-root .comparison-card").first()).toContainText("Metadata only");
  await expect(page.locator("#comparison-root .comparison-card").last()).toContainText("overlay p6");
  await expect(page.locator("#comparison-root")).toContainText("/4");
  await page.getByRole("button", { name: "Next turn", exact: true }).click();
  await expect(page.locator("#comparison-root")).toContainText("Turn 2 / 5");
  await page.locator("#comparison-study").selectOption("mechanistic");
  await page.locator("#comparison-case").selectOption("book-order-numeral");
  await expect(page.locator("#comparison-root")).toContainText("not a deployable repair");
  await expect(page.locator("#comparison-root .comparison-card").last()).toContainText("שני ספרים");
  await expect(page.getByRole("button", { name: "Run live baseline vs historical overlay" })).toBeDisabled();
  await page.getByLabel("Hide arm identities").check();
  await expect(page.locator("#comparison-root .comparison-card").first().getByRole("heading")).toHaveText("Output A");
});

test("comparison layout fits a mobile viewport and recorded navigation needs no GPU", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/research.html");
  await expect(page.locator("#comparison-status")).toContainText("recorded comparisons");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.locator("#comparison-case").selectOption("dev-market-apples-01:B2:47");
  await expect(page.locator("#comparison-root")).toContainText("seed 47");
  await expect(page.locator("#run")).toBeDisabled();
});

class Store implements ArtifactStore {
  private readonly rows = new Map<string, Artifact>();
  async get(id: string) { return this.rows.get(id); }
  async list() { return [...this.rows.values()]; }
  async put(a: Artifact) { this.rows.set(a.id, structuredClone(a)); }
}
test("paired live replay uses the same seeds and baseline route, preserves own histories and checkpoints turns", async () => {
  const prompts: string[] = [];
  const evaluations: GemmaResearchEvaluationRequest[] = [];
  const records = new Store();
  const bench = new ResearchWorkbench({
    destroy() {},
    researchTokenize(messages) {
      prompts.push(JSON.stringify(messages));
      return { promptTokenIds: [2, 20], continuationTokenIds: [], pieces: ["a", "b"] };
    },
    researchInspectWeights: async () => { throw new Error("Unused"); },
    researchEvaluate: async (request): Promise<GemmaResearchEvaluationResult> => {
      evaluations.push(request);
      const text = request.overlays?.length ? "edited" : "base";
      request.onProgress?.({ evaluatedPositions: 2, generatedTokens: 1 });
      return {
        promptTokenIds: [2, 20], tokenIds: [30], tokenLogProbs: [-0.5], text,
        stopReason: "end-token", captures: [], firstDecision: [], evaluatedPositions: 2, milliseconds: 1,
      };
    },
  }, records, { runId: "paired", revision: "fixture", checkpointRevision: "pin", checkpointSha256: "a".repeat(64), environment: "unit mock" });
  const request: PairedReplayRequest = {
    id: "two-turns", op: "pairedReplay", protocol: "conversation", messages: [{ role: "system", content: "fixed" }],
    playerTurns: ["one", "two"], seed: 23, maxNewTokens: 4, temperature: 0,
    arms: [
      { id: "base", label: "Baseline" },
      { id: "edited", label: "Historical steering", overlays: [{ layer: 20, a: Array(1536).fill(0), b: Array(1536).fill(0), coefficient: 0.1 }] },
    ],
  };
  const artifact = await bench.run(request);
  expect(artifact.status).toBe("completed");
  expect(prompts[0]).toBe(prompts[1]);
  expect(prompts[2]).toContain("base");
  expect(prompts[2]).not.toContain("edited");
  expect(prompts[3]).toContain("edited");
  expect(evaluations.map((e) => e.generate?.seed)).toEqual([23, 23, 24, 24]);
  expect(evaluations.map((e) => e.overlays?.length)).toEqual([0, 1, 0, 1]);
  expect(bench.usage.episodes).toBe(4);
  expect(bench.usage.generatedTokens).toBe(4);
  await bench.close();
});

test("numeric observations reject embedded amounts without claiming semantic grounding", () => {
  for (const text of ["148 שקלים", "48.5 שקלים", "48,5 שקלים", "-48 שקלים", ".48 שקלים", "1,048 שקלים"]) {
    expect(hasNumericAmount(text, 48), text).toBe(false);
  }
  for (const text of ["48 שקלים.", "המחיר 48.00 שקלים", "48,0"]) expect(hasNumericAmount(text, 48)).toBe(true);
  expect(() => hasNumericAmount("48", NaN)).toThrow("finite");
});

test("recorded metadata matches every level and absent intervention arms are explicit", async ({ page }) => {
  const catalogue = JSON.parse(readFileSync("public/research/comparisons.json", "utf8")) as {
    examples: { experiment: string; level: string; metadata: { cefr_level?: string } }[];
  };
  for (const example of catalogue.examples.filter((e) => e.experiment === "cefr")) {
    expect(example.metadata.cefr_level).toBe(example.level);
  }
  await page.goto("/research.html");
  await page.locator("#comparison-study").selectOption("mechanistic");
  await page.locator("#comparison-case").selectOption("shirt-order-numeral");
  expect(await page.locator("#comparison-before").inputValue()).not.toBe(await page.locator("#comparison-after").inputValue());
  await expect(page.locator("#comparison-root")).toContainText("No intervention recording is available");
  await expect(page.locator("#comparison-root")).toContainText("not a same-input intervention pair");
});

test("evidence dashboard exposes paired harm, uncertainty, control distribution and actual transfer", async ({ page }) => {
  await page.goto("/research.html");
  await expect(page.locator("#paired-statistics .outcome-counts > div")).toHaveText(["0Fixed", "6Broken", "0Still correct", "18Still wrong", "0Unknown"]);
  await expect(page.locator(".effect-interval")).toHaveCount(4);
  await expect(page.locator("#judge-audit")).toContainText("6/6 consistent");
  await expect(page.locator(".component-cell")).toHaveCount(18);
  await page.locator('.component-cell[data-case="shirt-order-numeral"][data-layer="34"][data-stage="attention"]').click();
  await expect(page.locator("#component-preview")).toContainText("שתי חולצות יהיו בסך 110 שקלים.");
  await expect(page.locator("#random-controls .evidence-dot")).toHaveCount(16);
  await expect(page.locator("#random-controls")).toContainText("empirical p = 0.0625");
  await expect(page.locator("#random-controls")).toContainText("שתי גילידים");
  await expect(page.locator("#component-evidence")).toContainText("not semantic or full-panel passes");
});

test("missing evidence is an explicit error and does not hide other panels", async ({ page }) => {
  await page.route("**/research/paired-statistics.json", (route) => route.fulfill({ status: 500, body: "unavailable" }));
  await page.goto("/research.html");
  await expect(page.locator("#paired-statistics [role=status]")).toContainText("HTTP 500");
  await expect(page.locator(".component-cell")).toHaveCount(18);
});

test("capture inspector keeps the selected boundary and never offers layer-input as a patch site", async ({ page }) => {
  await page.goto("/research.html");
  await page.locator("#import").setInputFiles({
    name: "captures.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify([{
      schemaVersion: 1, id: "fixture:captures", status: "completed", request: { id: "captures", op: "capture" },
      result: { captures: ["layer-input", "attention", "mlp", "layer-output"].map((stage) => ({
        site: { residual: 20 }, stage, tokenPosition: 4, values: [1, 2],
      })) },
    }])),
  });
  await page.locator("#ledger button").click();
  await expect(page.locator("#captures")).toContainText("layer-input");
  await expect(page.locator("#captures button")).toHaveCount(3);
  await page.getByRole("button", { name: "Use layer 20 attention as diagnostic donor", exact: true }).click();
  expect(JSON.parse(await page.locator("#request").inputValue()).stage).toBe("attention");
});
