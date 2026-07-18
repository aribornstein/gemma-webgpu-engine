import { expect, test } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  renderE2BPerformanceProofMarkdown,
  type E2BCaseRuntimeResult,
  type E2BMetricDistribution,
  type E2BPerformanceCase,
  type E2BPerformanceProofArtifact,
} from "../src/benchmark/e2b-performance-proof";

const LIVE_BENCHMARK_ENABLED = process.env.E2B_PERFORMANCE_PROOF === "1";
const CASE_IDS = [
  "short-greeting",
  "arithmetic",
  "arabic",
  "longer-instruction",
  "prefill-32-boundary",
] as const;

test.skip(!LIVE_BENCHMARK_ENABLED, "Set E2B_PERFORMANCE_PROOF=1 to run multi-GB live benchmarks");
test.setTimeout(30 * 60_000);

test("writes the isolated E2B performance proof and Markdown report", async ({ browser }) => {
  const ownedContext = await browser.newContext();
  const ownedPage = await ownedContext.newPage();
  await ownedPage.goto("/");
  const ownedArtifacts = [];
  for (const caseId of CASE_IDS) {
    ownedArtifacts.push(await ownedPage.evaluate(async (selectedCaseId) => {
      const modulePath = "/src/runtime/gemma-benchmark.ts";
      const { benchmarkGemmaGeneration } = await import(modulePath);
      return benchmarkGemmaGeneration({
        caseId: selectedCaseId,
        warmupIterations: 1,
        iterations: 3,
        prefillStrategy: "auto",
        sourceUrl: "/models/gemma-4-e2b/model.safetensors",
      });
    }, caseId));
  }
  await ownedContext.close();

  const transformersContext = await browser.newContext();
  const transformersPage = await transformersContext.newPage();
  await transformersPage.goto("/");
  const transformersArtifact = await transformersPage.evaluate(async (cases) => {
    const modulePath = "/src/benchmark/transformers-js-e2b.ts";
    const { benchmarkTransformersJsE2B } = await import(modulePath);
    return benchmarkTransformersJsE2B({
      cases,
      warmupIterations: 1,
      iterations: 3,
    });
  }, ownedArtifacts.map((artifact) => ({
    id: artifact.configuration.caseId,
    prompt: artifact.configuration.prompt,
    maxOutputTokens: artifact.configuration.maxNewTokens,
    expectedText: artifact.correctness.expectedText,
  })));
  await transformersContext.close();

  const liteRtContext = await browser.newContext();
  const liteRtPage = await liteRtContext.newPage();
  await liteRtPage.goto("/");
  const liteRtArtifact = await liteRtPage.evaluate(async (cases) => {
    const modulePath = "/src/benchmark/litert-lm-e2b.ts";
    const { benchmarkLiteRtLmE2B } = await import(modulePath);
    return benchmarkLiteRtLmE2B({
      cases,
      contextCapacity: 2048,
      warmupIterations: 1,
      iterations: 3,
    });
  }, ownedArtifacts.map((artifact) => ({
    id: artifact.configuration.caseId,
    prompt: artifact.configuration.prompt,
    maxOutputTokens: artifact.configuration.maxNewTokens,
    expectedText: artifact.correctness.expectedText,
  })));
  await liteRtContext.close();

  const priorPath = path.join(
    process.cwd(),
    "benchmarks/huggingface-same-device-gap-analysis.electron-148.json",
  );
  const priorHuggingFace = JSON.parse(await readFile(priorPath, "utf8"));
  const artifact = createProofArtifact(
    ownedArtifacts,
    transformersArtifact,
    liteRtArtifact,
    priorHuggingFace,
  );
  const rawPath = path.join(process.cwd(), "benchmarks/e2b-performance-proof.chrome.json");
  const reportPath = path.join(process.cwd(), "benchmarks/BENCHMARK_RESULTS.md");
  await writeFile(rawPath, `${JSON.stringify({ artifact, raw: {
    owned: ownedArtifacts,
    transformersJs: transformersArtifact,
    liteRtLm: liteRtArtifact,
    priorHuggingFaceSource: path.basename(priorPath),
  } }, null, 2)}\n`);
  await writeFile(reportPath, renderE2BPerformanceProofMarkdown(artifact));

  expect(artifact.status).toBe("partial");
  expect(artifact.cases).toHaveLength(CASE_IDS.length);
  expect(artifact.cases.every((benchmarkCase) =>
    benchmarkCase.results.some((result) => result.runtimeId === "owned-webgpu") &&
    benchmarkCase.results.some((result) => result.runtimeId === "transformers-js") &&
    benchmarkCase.results.some((result) => result.runtimeId === "litert-lm-web")
  )).toBe(true);
});

function createProofArtifact(
  ownedArtifacts: readonly any[],
  transformersArtifact: any,
  liteRtArtifact: any,
  priorHuggingFace: any,
): E2BPerformanceProofArtifact {
  const firstOwned = ownedArtifacts[0];
  return {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    status: "partial",
    model: {
      id: firstOwned.model.id,
      revision: firstOwned.model.revision,
    },
    environment: {
      userAgent: firstOwned.environment.userAgent,
      adapter: Object.values(firstOwned.environment.adapterInfo).join(" / ") || "Unavailable",
      device: "MacBook Air (local benchmark host)",
    },
    methodology: [
      "Owned WebGPU, Transformers.js, and LiteRT-LM ran sequentially in fresh browser contexts; every runtime-owned session, conversation, engine, and context was destroyed before the next runtime.",
      "Each live case used one warmup followed by three measured greedy generations with a fresh prompt cache or conversation.",
      "Externally observed TTFT and total wall time are primary. Runtime-native counters are supplemental unless their timing boundaries are demonstrably equivalent.",
      "The pinned Hugging Face row is prior same-device evidence from Chrome 148/Electron 42 and is not treated as a current-browser measurement.",
      "Transformers.js uses the pinned ONNX Community q4f16 text-only export. It is a model-family comparison, not file-identical model execution.",
      "LiteRT-LM uses Google's specially optimized text-only Web .litertlm artifact. It is a model-family comparison, not file-identical model execution.",
      "No browser runtime exposes a portable retained GPU-memory measurement with equivalent boundaries; memory is therefore omitted from same-device speedup gates.",
    ],
    runtimes: [
      {
        id: "owned-webgpu",
        label: "Owned WebGPU",
        version: "workspace",
        modelArtifact: `${firstOwned.model.sourceKey} (${firstOwned.model.fileSize} bytes)`,
        artifactEquivalence: "pinned-source-equivalent",
        status: "same-device-measured",
        notes: [
          "Loads the pinned mobile-QAT safetensors artifact through repository-owned kernels.",
          `Fresh session load from the local pinned file: ${firstOwned.load.sessionLoadMs} ms.`,
          `Retained GPU buffers for the first case: ${firstOwned.memory.retainedGpuBufferBytes} bytes across ${firstOwned.memory.retainedGpuBufferCount} buffers; driver and pipeline allocations are excluded.`,
        ],
      },
      {
        id: "pinned-hugging-face-webgpu",
        label: "Pinned Hugging Face WebGPU",
        version: priorHuggingFace.huggingFaceRuntime.sourceCommit,
        modelArtifact: priorHuggingFace.model.sourceKey,
        artifactEquivalence: "pinned-source-equivalent",
        status: "prior-same-device-measured",
        notes: [
          "Reuses the existing exact-output same-device artifact; it was not rerun because its vendored upstream bundle is not present in this workspace.",
          `Previous clean cached load: ${priorHuggingFace.load.cleanCachedLoadMs} ms.`,
        ],
      },
      {
        id: "transformers-js",
        label: "Transformers.js",
        version: transformersArtifact.runtimeVersion,
        modelArtifact: `${transformersArtifact.modelId}@${transformersArtifact.modelRevision} (${transformersArtifact.modelVariant})`,
        artifactEquivalence: "model-family-only",
        status: "same-device-measured",
        notes: [
          `Fresh processor and model load: ${transformersArtifact.loadMs} ms.`,
          ...transformersArtifact.limitations,
        ],
      },
      {
        id: "litert-lm-web",
        label: "LiteRT-LM Web",
        version: liteRtArtifact.runtimeVersion,
        modelArtifact: liteRtArtifact.modelUrl,
        artifactEquivalence: "model-family-only",
        status: "same-device-measured",
        notes: [
          `Fresh engine load: ${liteRtArtifact.loadMs} ms.`,
          ...liteRtArtifact.limitations,
        ],
      },
    ],
    cases: ownedArtifacts.map((ownedArtifact) => createCase(
      ownedArtifact,
      transformersArtifact,
      liteRtArtifact,
      priorHuggingFace,
    )),
    publishedReferences: [{
      runtimeId: "litert-lm-web",
      device: "MacBook Pro M4 Max / WebGPU",
      promptTokens: 1024,
      outputTokens: 256,
      prefillTokensPerSecond: 4853,
      decodeTokensPerSecond: 73,
      timeToFirstTokenSeconds: 1.09,
      modelSizeMb: 2008,
      memoryMb: 1800,
      source: "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm",
    }],
  };
}

function createCase(
  ownedArtifact: any,
  transformersArtifact: any,
  liteRtArtifact: any,
  priorHuggingFace: any,
): E2BPerformanceCase {
  const caseId = ownedArtifact.configuration.caseId;
  const priorCase = priorHuggingFace.canonical.cases.find((candidate: any) =>
    candidate.id === caseId
  );
  const liteRtCase = liteRtArtifact.cases.find((candidate: any) => candidate.id === caseId);
  const transformersCase = transformersArtifact.cases.find((candidate: any) =>
    candidate.id === caseId
  );
  const results: E2BCaseRuntimeResult[] = [
    {
      runtimeId: "owned-webgpu",
      evidenceStatus: "same-device-measured",
      exactOutputMatch: ownedArtifact.correctness.allIterationsMatchGolden,
      generatedTokens: ownedArtifact.configuration.expectedOutputTokens,
      metrics: {
        ttftMs: fromOwnedDistribution(ownedArtifact.summary.timeToFirstToken),
        itlMs: fromOwnedDistribution(ownedArtifact.summary.interTokenLatency),
        tpotMs: fromOwnedDistribution(ownedArtifact.summary.timePerOutputToken),
        decodeTokensPerSecond: ownedArtifact.summary.warmDecodeTokensPerSecond,
        totalMs: fromOwnedDistribution(ownedArtifact.summary.total),
      },
    },
  ];
  if (priorCase) results.push(priorHuggingFaceResult(priorCase));
  results.push(transformersJsResult(transformersCase));
  results.push(liteRtResult(liteRtCase));
  return {
    id: caseId,
    promptTokens: ownedArtifact.configuration.promptTokens,
    expectedOutputTokens: ownedArtifact.configuration.expectedOutputTokens,
    results,
  };
}

function transformersJsResult(testCase: any): E2BCaseRuntimeResult {
  const samples = testCase.samples;
  return {
    runtimeId: "transformers-js",
    evidenceStatus: "same-device-measured",
    exactOutputMatch: samples.every((sample: any) => sample.exactTextMatch),
    generatedTokens: median(samples.map((sample: any) =>
      sample.generatedTokenIds.length
    )),
    metrics: {
      ttftMs: distribution(samples.flatMap((sample: any) =>
        sample.timing.timeToFirstTokenMs === null
          ? []
          : [sample.timing.timeToFirstTokenMs]
      )),
      itlMs: distribution(samples.flatMap((sample: any) =>
        sample.timing.interTokenLatencyMs
      )),
      tpotMs: distribution(samples.flatMap((sample: any) =>
        sample.timing.timePerOutputTokenMs === null
          ? []
          : [sample.timing.timePerOutputTokenMs]
      )),
      decodeTokensPerSecond: median(samples.flatMap((sample: any) =>
        sample.timing.decodeTokensPerSecond === null
          ? []
          : [sample.timing.decodeTokensPerSecond]
      )),
      totalMs: distribution(samples.map((sample: any) => sample.timing.totalMs)),
    },
    note: "Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.",
  };
}

function priorHuggingFaceResult(priorCase: any): E2BCaseRuntimeResult {
  const value = priorCase.huggingFace;
  return {
    runtimeId: "pinned-hugging-face-webgpu",
    evidenceStatus: "prior-same-device-measured",
    exactOutputMatch: true,
    generatedTokens: priorCase.generatedTokenIds.length,
    metrics: {
      ttftMs: pair(value.ttftMedianMs, value.ttftP95Ms),
      itlMs: pair(value.itlMedianMs, value.itlP95Ms),
      tpotMs: value.steadyTokensPerSecond
        ? pair(1000 / value.steadyTokensPerSecond, 1000 / value.steadyTokensPerSecond)
        : null,
      decodeTokensPerSecond: value.steadyTokensPerSecond ?? null,
      totalMs: pair(value.totalMedianMs, value.totalP95Ms),
    },
    note: "Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.",
  };
}

function liteRtResult(testCase: any): E2BCaseRuntimeResult {
  const samples = testCase.samples;
  const nativeTtftMs = median(samples.map((sample: any) =>
    sample.native.timeToFirstTokenMs
  ));
  return {
    runtimeId: "litert-lm-web",
    evidenceStatus: "same-device-measured",
    exactOutputMatch: samples.every((sample: any) => sample.exactTextMatch),
    generatedTokens: median(samples.map((sample: any) => sample.native.decodeTokenCount)),
    metrics: {
      ttftMs: distribution(samples.flatMap((sample: any) =>
        sample.externallyObserved.timeToFirstChunkMs === null
          ? []
          : [sample.externallyObserved.timeToFirstChunkMs]
      )),
      itlMs: distribution(samples.flatMap((sample: any) =>
        sample.externallyObserved.chunkIntervalMs
      )),
      tpotMs: distribution(samples.flatMap((sample: any) =>
        sample.native.decodeTokensPerSecond > 0
          ? [1000 / sample.native.decodeTokensPerSecond]
          : []
      )),
      decodeTokensPerSecond: median(samples.map((sample: any) =>
        sample.native.decodeTokensPerSecond
      )),
      totalMs: distribution(samples.map((sample: any) => sample.externallyObserved.totalMs)),
    },
    note: `ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is ${nativeTtftMs} ms.`,
  };
}

function fromOwnedDistribution(value: any): E2BMetricDistribution | null {
  return value === null ? null : { median: value.medianMs, p95: value.p95Ms };
}

function pair(medianValue: number | undefined, p95Value: number | undefined): E2BMetricDistribution | null {
  return medianValue === undefined || p95Value === undefined
    ? null
    : { median: round(medianValue), p95: round(p95Value) };
}

function distribution(values: readonly number[]): E2BMetricDistribution | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return { median: round(percentile(sorted, 0.5)), p95: round(percentile(sorted, 0.95)) };
}

function median(values: readonly number[]): number | null {
  return distribution(values)?.median ?? null;
}

function percentile(sorted: readonly number[], quantile: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * quantile) - 1))];
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}