import { GEMMA_4_E2B_CACHE_SPEC } from "../model/cached-safetensors";
import {
  loadGemmaGenerationSession,
  type GemmaGenerationTiming,
  type GemmaSessionLoadOptions,
} from "./gemma-session";
import {
  GEMMA_GREEDY_GOLDEN_CASES,
  type GemmaGreedyGoldenCase,
} from "./gemma-golden";
import { calculateGemmaGenerationThroughput } from "./generation-throughput";

const MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers";
const MODEL_REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";
const DEFAULT_CASE_ID = "longer-instruction";
const DEFAULT_WARMUP_ITERATIONS = 1;
const DEFAULT_ITERATIONS = 3;

export interface GemmaGenerationBenchmarkOptions extends GemmaSessionLoadOptions {
  caseId?: string;
  warmupIterations?: number;
  iterations?: number;
}

export interface GemmaGenerationBenchmarkSample {
  iteration: number;
  timing: GemmaGenerationTiming;
  generatedTokenIds: readonly number[];
  text: string;
  stopReason: string;
  exactGoldenMatch: boolean;
}

export interface GemmaBenchmarkDistribution {
  medianMs: number;
  p95Ms: number;
  averageMs: number;
}

export interface GemmaGenerationBenchmarkSummary {
  prefill: GemmaBenchmarkDistribution;
  timeToFirstToken: GemmaBenchmarkDistribution;
  decodeToken: GemmaBenchmarkDistribution;
  total: GemmaBenchmarkDistribution;
  warmDecodeTokensPerSecond: number;
  endToEndTokensPerSecond: number;
}

export interface GemmaGenerationBenchmarkArtifact {
  schemaVersion: 1;
  capturedAt: string;
  status: "owned-full-generation-baseline";
  speedupClaim: null;
  model: {
    id: string;
    revision: string;
    sourceKey: string;
    fileSize: number;
  };
  environment: {
    userAgent: string;
    adapterInfo: Record<string, string>;
    powerPreference: "high-performance";
  };
  configuration: {
    caseId: string;
    prompt: string;
    promptTokens: number;
    expectedOutputTokens: number;
    maxNewTokens: number;
    cacheCapacity: number;
    prefillStrategy: "auto" | "fixed-32" | "chunked-32" | "sequential";
    warmupIterations: number;
    iterations: number;
  };
  correctness: {
    expectedTokenIds: readonly number[];
    expectedText: string;
    allIterationsMatchGolden: boolean;
  };
  load: {
    sessionLoadMs: number;
    note: string;
  };
  memory: {
    retainedGpuBufferCount: number;
    retainedGpuBufferBytes: number;
    scope: "retained-resource-graph";
    note: string;
  };
  samples: readonly GemmaGenerationBenchmarkSample[];
  summary: GemmaGenerationBenchmarkSummary;
}

export async function benchmarkGemmaGeneration(
  options: GemmaGenerationBenchmarkOptions = {},
): Promise<GemmaGenerationBenchmarkArtifact> {
  const testCase = findGoldenCase(options.caseId ?? DEFAULT_CASE_ID);
  const warmupIterations = validateIterationCount(
    options.warmupIterations ?? DEFAULT_WARMUP_ITERATIONS,
    "warmup",
    true,
  );
  const iterations = validateIterationCount(
    options.iterations ?? DEFAULT_ITERATIONS,
    "measured",
    false,
  );
  const cacheCapacity = options.cacheCapacity ?? 512;
  const prefillStrategy = options.prefillStrategy ?? "auto";
  const adapterInfoPromise = readAdapterInfo();
  const loadStartedAt = performance.now();
  const session = await loadGemmaGenerationSession({ cacheCapacity, prefillStrategy });
  const sessionLoadMs = performance.now() - loadStartedAt;
  const memory = session.estimateRetainedGpuMemory();

  try {
    for (let iteration = 0; iteration < warmupIterations; iteration += 1) {
      const result = await session.generate(testCase.prompt, {
        maxNewTokens: testCase.maxNewTokens,
      });
      assertGoldenMatch(testCase, result.generatedTokenIds, result.text, result.stoppedOnEndToken);
    }

    const samples: GemmaGenerationBenchmarkSample[] = [];
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const measured = await session.generateMeasured(testCase.prompt, {
        maxNewTokens: testCase.maxNewTokens,
      });
      const exactGoldenMatch = matchesGolden(
        testCase,
        measured.result.generatedTokenIds,
        measured.result.text,
        measured.result.stoppedOnEndToken,
      );
      if (!exactGoldenMatch) {
        assertGoldenMatch(
          testCase,
          measured.result.generatedTokenIds,
          measured.result.text,
          measured.result.stoppedOnEndToken,
        );
      }
      samples.push({
        iteration,
        timing: roundTiming(measured.timing),
        generatedTokenIds: Object.freeze([...measured.result.generatedTokenIds]),
        text: measured.result.text,
        stopReason: measured.result.stopReason,
        exactGoldenMatch,
      });
    }

    return {
      schemaVersion: 1,
      capturedAt: new Date().toISOString(),
      status: "owned-full-generation-baseline",
      speedupClaim: null,
      model: {
        id: MODEL_ID,
        revision: MODEL_REVISION,
        sourceKey: GEMMA_4_E2B_CACHE_SPEC.sourceKey,
        fileSize: GEMMA_4_E2B_CACHE_SPEC.fileSize,
      },
      environment: {
        userAgent: navigator.userAgent,
        adapterInfo: await adapterInfoPromise,
        powerPreference: "high-performance",
      },
      configuration: {
        caseId: testCase.id,
        prompt: testCase.prompt,
        promptTokens: testCase.promptTokenIds.length,
        expectedOutputTokens: testCase.generatedTokenIds.length,
        maxNewTokens: testCase.maxNewTokens,
        cacheCapacity: session.cacheCapacity,
        prefillStrategy,
        warmupIterations,
        iterations,
      },
      correctness: {
        expectedTokenIds: testCase.generatedTokenIds,
        expectedText: testCase.text,
        allIterationsMatchGolden: samples.every((sample) => sample.exactGoldenMatch),
      },
      load: {
        sessionLoadMs: round(sessionLoadMs),
        note: prefillStrategy !== "fixed-32"
          ? "Includes device/cache/tokenizer initialization, model resource loading, and pipeline creation in the current browser process. Automatic routing currently selects the measured sequential reference path."
          : "Includes device/cache/tokenizer initialization, model resource loading, pipeline creation, and fixed-32 prefill resource creation in the current browser process.",
      },
      memory: {
        retainedGpuBufferCount: memory.gpuBufferCount,
        retainedGpuBufferBytes: memory.gpuBufferBytes,
        scope: memory.scope,
        note: "Deduplicated byte sum of GPUBuffer objects retained by the composed decode graph, optional prefill graph, and logits readback buffer. Driver and pipeline memory are excluded.",
      },
      samples,
      summary: summarizeGemmaBenchmarkSamples(samples),
    };
  } finally {
    session.destroy();
  }
}

export function summarizeGemmaBenchmarkSamples(
  samples: readonly GemmaGenerationBenchmarkSample[],
): GemmaGenerationBenchmarkSummary {
  if (samples.length === 0) throw new Error("Gemma benchmark requires at least one sample");
  const decodeSamples = samples.flatMap((sample) => sample.timing.decodeTokenMs);
  if (decodeSamples.length === 0) {
    throw new Error("Gemma benchmark requires at least one measured decode step");
  }
  const totalGeneratedTokens = samples.reduce(
    (sum, sample) => sum + sample.generatedTokenIds.length,
    0,
  );
  const throughput = calculateGemmaGenerationThroughput(
    samples.map((sample) => sample.timing),
    totalGeneratedTokens,
  );
  return {
    prefill: distribution(samples.map((sample) => sample.timing.prefillMs)),
    timeToFirstToken: distribution(
      samples.map((sample) => sample.timing.timeToFirstTokenMs),
    ),
    decodeToken: distribution(decodeSamples),
    total: distribution(samples.map((sample) => sample.timing.totalMs)),
    warmDecodeTokensPerSecond: round(throughput.warmDecodeTokensPerSecond!),
    endToEndTokensPerSecond: round(throughput.endToEndTokensPerSecond!),
  };
}

function findGoldenCase(caseId: string): GemmaGreedyGoldenCase {
  const testCase = GEMMA_GREEDY_GOLDEN_CASES.find((candidate) => candidate.id === caseId);
  if (!testCase) throw new Error(`Unknown Gemma benchmark golden case: ${caseId}`);
  return testCase;
}

function validateIterationCount(value: number, label: string, allowZero: boolean): number {
  const minimum = allowZero ? 0 : 1;
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`Gemma ${label} iterations must be an integer >= ${minimum}`);
  }
  return value;
}

function assertGoldenMatch(
  testCase: GemmaGreedyGoldenCase,
  generatedTokenIds: readonly number[],
  text: string,
  stoppedOnEndToken: boolean,
): void {
  if (matchesGolden(testCase, generatedTokenIds, text, stoppedOnEndToken)) return;
  throw new Error(
    `Gemma benchmark correctness gate failed for ${testCase.id}: ` +
    `expected ${JSON.stringify(testCase.generatedTokenIds)} / ${JSON.stringify(testCase.text)}, ` +
    `received ${JSON.stringify(generatedTokenIds)} / ${JSON.stringify(text)}`,
  );
}

function matchesGolden(
  testCase: GemmaGreedyGoldenCase,
  generatedTokenIds: readonly number[],
  text: string,
  stoppedOnEndToken: boolean,
): boolean {
  return stoppedOnEndToken === testCase.stoppedOnEndToken &&
    text === testCase.text &&
    generatedTokenIds.length === testCase.generatedTokenIds.length &&
    generatedTokenIds.every((tokenId, index) => tokenId === testCase.generatedTokenIds[index]);
}

function distribution(values: readonly number[]): GemmaBenchmarkDistribution {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    medianMs: round(percentile(sorted, 0.5)),
    p95Ms: round(percentile(sorted, 0.95)),
    averageMs: round(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
  };
}

function percentile(sortedValues: readonly number[], quantile: number): number {
  const index = Math.min(
    sortedValues.length - 1,
    Math.max(0, Math.ceil(sortedValues.length * quantile) - 1),
  );
  return sortedValues[index];
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function roundTiming(timing: GemmaGenerationTiming): GemmaGenerationTiming {
  return {
    requestSetupMs: round(timing.requestSetupMs),
    visionPreprocessMs: round(timing.visionPreprocessMs),
    visionEncodeMs: round(timing.visionEncodeMs),
    cacheResetMs: round(timing.cacheResetMs),
    promptTokensReused: timing.promptTokensReused,
    prefillMs: round(timing.prefillMs),
    prefillMode: timing.prefillMode,
    timeToFirstTokenMs: round(timing.timeToFirstTokenMs),
    decodeTokenMs: Object.freeze(timing.decodeTokenMs.map(round)),
    logitsReadbackMs: round(timing.logitsReadbackMs),
    callbackMs: round(timing.callbackMs),
    totalMs: round(timing.totalMs),
  };
}

async function readAdapterInfo(): Promise<Record<string, string>> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return {};
  const info = adapter.info;
  return Object.fromEntries(
    ["vendor", "architecture", "device", "description"].flatMap((key) => {
      const value = info[key as keyof GPUAdapterInfo];
      return typeof value === "string" && value.length > 0 ? [[key, value]] : [];
    }),
  );
}
