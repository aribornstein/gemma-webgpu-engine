import { expect, test } from "@playwright/test";
import { checkGrounding, checkVocabulary, promotionVerdict, validateAssessment, validateWorld, type JudgeAssessment, type WorldContract } from "../src/research/workbench/contracts";
import { DurableArtifactStore, ResearchWorkbench, hashJson, type Artifact, type ArtifactStore, type ResearchProvenance } from "../src/research/workbench/runner";
import { inspectWeightSlice, validateWeightSliceRequest } from "../src/research/workbench/weights";
import { gemmaLayerTensorContracts } from "../src/model/gemma-layer-plan";
import type { CachedTensorDescriptor } from "../src/model/cached-safetensors";
import type { GemmaResearchEvaluationRequest, GemmaResearchEvaluationResult } from "../src/research/engine-types";

const world: WorldContract = {
  currency: "ILS", quantity: 2, unitPrice: 5, totalPrice: 10, paid: false,
  item: "apple", topic: "shopping", npcGoal: "Quote a price without claiming payment",
  vocabularyPolicy: "unrestricted-diagnostic",
};
const text = "שני תפוחים עולים עשרה שקלים.";
const assessment = (role: JudgeAssessment["role"], score = 9): JudgeAssessment => ({
  episodeId: "e", judgeId: `judge-${role}`, role, rubricVersion: "mechanistic-quality-1", abstain: false,
  criticalViolations: [],
  ratings: ({
    language: ["grammar", "naturalness"],
    task: ["vocabulary", "grounding", "goal"],
    "level-conversation": ["levelSuitability", "coherence"],
  } as const)[role].map((criterion) => ({ criterion, score, evidence: ["שני תפוחים"] })),
});

test("world contracts distinguish quantity, unit price, totals and currency", () => {
  expect(() => validateWorld({ ...world, totalPrice: 5 })).toThrow("quantity * unitPrice");
  expect(checkGrounding(world, text).status).toBe("unknown");
  expect(checkGrounding(world, text, [{ field: "totalPrice", value: 10, quote: "עשרה שקלים" }]).status).toBe("pass");
  expect(checkGrounding(world, text, [{ field: "unitPrice", value: 10, quote: "עשרה שקלים" }]).status).toBe("fail");
  expect(checkGrounding(world, text, [{ field: "currency", value: "EUR", quote: "שקלים" }]).status).toBe("fail");
  expect(() => checkGrounding(world, text, [{ field: "paid", value: true, quote: "not there" }])).toThrow("evidence");
  expect(checkVocabulary(world, text).status).toBe("unknown");
  expect(checkVocabulary({ ...world, vocabularyPolicy: "explicit-forms", allowedForms: ["שני", "תפוחים", "עולים", "עשרה", "שקלים"] }, text).status).toBe("pass");
});

test("nine-out-of-ten gates do not hide missing criteria, evidence or abstentions", () => {
  const panel = [assessment("language"), assessment("task"), assessment("level-conversation")];
  expect(promotionVerdict(panel, "e", text, ["pass"])).toBe("pass");
  expect(promotionVerdict([assessment("language", 8.99), ...panel.slice(1)], "e", text, ["pass"])).toBe("fail");
  expect(promotionVerdict(panel, "e", text, ["unknown"])).toBe("unknown");
  expect(promotionVerdict(panel.slice(0, 2), "e", text, ["pass"])).toBe("unknown");
  expect(() => validateAssessment({ ...panel[0], ratings: [] }, "e", text)).toThrow("Missing");
  expect(() => validateAssessment({ ...panel[0], ratings: [...panel[0].ratings, panel[0].ratings[0]] }, "e", text)).toThrow("duplicate");
  expect(() => validateAssessment({ ...panel[0], episodeId: "wrong" }, "e", text)).toThrow("episode");
  expect(() => validateAssessment({ ...panel[0], ratings: panel[0].ratings.map((r) => ({ ...r, evidence: ["invented"] })) }, "e", text)).toThrow("evidence");
  expect(promotionVerdict([{ ...panel[0], abstain: true, ratings: panel[0].ratings.map((r) => ({ ...r, score: null, evidence: [] })) }, ...panel.slice(1)], "e", text, ["pass"])).toBe("unknown");
});

test("weight inspection is bounded and decodes int2 codes with exact per-row scales", async () => {
  const contracts = gemmaLayerTensorContracts(20);
  const descriptors = new Map<string, CachedTensorDescriptor>();
  for (const contract of contracts) {
    const byteLength = contract.shape.reduce((a, b) => a * b, 1) * (contract.dtype === "F32" ? 4 : contract.dtype === "BF16" ? 2 : 1);
    descriptors.set(contract.name, { ...contract, byteLength, begin: 0, end: byteLength });
  }
  const reads: { name: string; offset: number; length: number }[] = [];
  const slice = await inspectWeightSlice({
    descriptors,
    readTensors: async () => { throw new Error("Inspection must not read whole tensors"); },
    readTensorSlice: async (name, offset, length) => {
      reads.push({ name, offset, length });
      if (name.endsWith(".weight")) return new Uint8Array(length).fill(0b11100100);
      const bytes = new Uint8Array(length);
      const view = new DataView(bytes.buffer);
      for (let i = 0; i < length; i += 4) view.setFloat32(i, 0.5, true);
      return bytes;
    },
  }, { layer: 20, projection: "down", rowStart: 2, rowCount: 1, columnStart: 1, columnCount: 4 });
  expect(slice.codes).toEqual([-1, 0, 1, -2]);
  expect(slice.values).toEqual([-0.5, 0, 0.5, -1]);
  expect(reads.every((read) => read.length <= 4)).toBe(true);
  expect(() => validateWeightSliceRequest({ layer: 20, projection: "down", rowStart: 0, rowCount: 65, columnStart: 0, columnCount: 32 })).toThrow("limited");
});

class MemoryStore implements ArtifactStore {
  readonly data = new Map<string, Artifact>();
  async get(id: string) { return this.data.get(id); }
  async put(artifact: Artifact) { this.data.set(artifact.id, structuredClone(artifact)); }
  async list() { return [...this.data.values()]; }
}
const provenance: ResearchProvenance = {
  runId: "test", revision: "sourcehash", checkpointRevision: "pin", checkpointSha256: "a".repeat(64), environment: "fake-unit-backend",
};
function mockBackend() {
  let calls = 0;
  return {
    get calls() { return calls; },
    destroy() {},
    researchTokenize() { return { promptTokenIds: [2, 10], continuationTokenIds: [20], pieces: ["a", "b", "c"] }; },
    researchInspectWeights: async () => { throw new Error("Unused in mock"); },
    researchEvaluate: async (request: GemmaResearchEvaluationRequest): Promise<GemmaResearchEvaluationResult> => {
      calls += 1;
      request.onProgress?.({ evaluatedPositions: 2, generatedTokens: request.generate ? 1 : 0 });
      return {
        promptTokenIds: [2, 10], tokenIds: [20], tokenLogProbs: [-0.5], text,
        stopReason: request.generate ? "end-token" : "forced", captures: [], firstDecision: [], evaluatedPositions: 2, milliseconds: 1,
      };
    },
  };
}
test("shared runner persists artifacts, resumes exact jobs and rejects mutations and budget excess", async () => {
  const backend = mockBackend();
  const store = new MemoryStore();
  const bench = new ResearchWorkbench(backend, store, provenance, { evaluatedPositions: 4, generatedTokens: 2, diagnostics: 3, episodes: 2 });
  const request = { id: "s1", op: "score" as const, messages: [{ role: "user" as const, content: "Hi" }], continuation: "שלום" };
  const first = await bench.run(request);
  expect(first.status).toBe("completed");
  expect(first.resultHash).toBe(await hashJson(first.result));
  await bench.run(request);
  expect(backend.calls).toBe(1);
  await expect(bench.run({ ...request, continuation: "changed" })).rejects.toThrow("reused");
  await bench.run({ ...request, id: "s2" });
  await expect(bench.run({ ...request, id: "s3" })).rejects.toThrow("budget");
  expect(backend.calls).toBe(2);
  expect((await store.get("test:s3"))?.status).toBe("failed");
  await bench.close();
});

test("workbench page opens without a model and exposes the same typed API to agents", async ({ page }) => {
  await page.goto("/research.html");
  await expect(page.getByRole("heading", { name: "Mechanism-first workbench" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run experiment" })).toBeDisabled();
  await expect(page.getByRole("status")).toContainText("No model loaded");
  await page.locator("#operation").selectOption("inspectWeights");
  await expect(page.locator("#request")).toHaveValue(/"projection": "down"/);
  expect(await page.evaluate(() => typeof window.researchWorkbench.run)).toBe("function");
  expect(await page.evaluate(() => typeof window.researchWorkbench.cancel)).toBe("function");
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

test("cancelling the active job preserves a paused queue; closing rejects queued work", async () => {
  const started = deferred<void>();
  const backend = mockBackend();
  const store = new MemoryStore();
  let calls = 0;
  const bench = new ResearchWorkbench({
    ...backend,
    researchEvaluate: async (request) => {
      calls += 1;
      started.resolve();
      await new Promise<void>((_, reject) => {
        request.signal?.addEventListener("abort", () => reject(request.signal?.reason), { once: true });
      });
      return backend.researchEvaluate(request);
    },
  }, store, provenance);
  const request = { id: "first", op: "replay" as const, messages: [{ role: "user" as const, content: "Hi" }], maxNewTokens: 10 };
  const first = bench.run(request).then(() => "unexpected completion", (error: Error) => error.name);
  await started.promise;
  bench.pause();
  const second = bench.run({ ...request, id: "second" }).then(() => "unexpected completion", (error: Error) => error.message);
  bench.cancel();
  expect(await first).toBe("AbortError");
  expect(bench.isPaused).toBe(true);
  expect(calls).toBe(1);
  await bench.close();
  expect(await second).toContain("closed");
  expect(calls).toBe(1);
});

test("durable reservations precede evaluation and survive a simulated lost browser", async () => {
  const disk = new MemoryStore();
  const local = new MemoryStore();
  const started = deferred<void>();
  const release = deferred<void>();
  const backend = mockBackend();
  const bench = new ResearchWorkbench({
    ...backend,
    researchEvaluate: async (request) => {
      const reservation = await disk.get("test:long");
      expect(reservation?.status).toBe("running");
      expect(reservation?.usage.evaluatedPositions).toBe(129);
      expect(reservation?.usage.generatedTokens).toBe(128);
      started.resolve();
      await release.promise;
      return backend.researchEvaluate(request);
    },
  }, new DurableArtifactStore(local, (a) => disk.put(a)), provenance);
  const request = { id: "long", op: "replay" as const, messages: [{ role: "user" as const, content: "Hi" }], maxNewTokens: 128 };
  const running = bench.run(request);
  await started.promise;
  const recoveredStore = new MemoryStore();
  for (const artifact of await disk.list()) await recoveredStore.put(artifact);
  const recovered = new ResearchWorkbench(mockBackend(), recoveredStore, provenance);
  await recovered.restoreUsage();
  expect(recovered.usage).toEqual({ evaluatedPositions: 129, generatedTokens: 128, diagnostics: 1, episodes: 1 });
  expect((await recoveredStore.get("test:long"))?.status).toBe("failed");
  await expect(recovered.run(request)).rejects.toThrow("explicit new attempt");
  release.resolve();
  await running;
  await Promise.all([bench.close(), recovered.close()]);
});

test("failed durable persistence prevents any model evaluation", async () => {
  const backend = mockBackend();
  const bench = new ResearchWorkbench(backend, new DurableArtifactStore(new MemoryStore(), async () => {
    throw new Error("Disk unavailable");
  }), provenance);
  await expect(bench.run({
    id: "disk-fail", op: "replay", messages: [{ role: "user", content: "Hi" }], maxNewTokens: 8,
  })).rejects.toThrow("Disk unavailable");
  expect(backend.calls).toBe(0);
  await bench.close();
});

test("close during model loading destroys the late session instead of enabling work", async ({ page }) => {
  await page.route("**/src/runtime/gemma-session.ts", (route) => route.fulfill({
    contentType: "text/javascript",
    body: `export async function loadGemmaGenerationSession() {
      window.testLoadStarted = true;
      await new Promise(resolve => { window.testReleaseLoad = resolve; });
      return { destroy() { window.testDestroyCount = (window.testDestroyCount || 0) + 1; } };
    }`,
  }));
  await page.goto("/research.html");
  await page.evaluate(() => {
    Object.assign(window, {
      testLoading: window.researchWorkbench.load().then(() => "ready", (error: Error) => error.name),
    });
  });
  await page.waitForFunction(() => "testLoadStarted" in window);
  await page.evaluate(() => { Object.assign(window, { testClosing: window.researchWorkbench.close() }); });
  const state = await page.evaluate(async () => {
    const release: unknown = Reflect.get(window, "testReleaseLoad");
    if (typeof release !== "function") throw new Error("Missing delayed loader");
    release();
    await Reflect.get(window, "testClosing");
    return { loading: await Reflect.get(window, "testLoading"), destroyed: Reflect.get(window, "testDestroyCount") };
  });
  expect(state).toEqual({ loading: "AbortError", destroyed: 1 });
  await expect(page.getByRole("button", { name: "Run experiment" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Load research session" })).toBeEnabled();
  await expect(page.getByRole("status")).toContainText("closed");
});
