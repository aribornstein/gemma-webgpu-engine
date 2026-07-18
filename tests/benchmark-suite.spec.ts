import { expect, test } from "@playwright/test";
import { measureGeneration } from "../src/benchmark/suite/measurement";
import { renderBenchmarkArtifacts } from "../src/benchmark/suite/report";
import { runScheduledCase } from "../src/benchmark/suite/runner";
import { createBenchmarkSchedule } from "../src/benchmark/suite/schedule";
import {
  bootstrapMedianConfidenceInterval,
  calculateStatistics,
  percentile,
} from "../src/benchmark/suite/statistics";
import { summarizeBenchmarkRuns } from "../src/benchmark/suite/summary";
import type {
  BenchmarkAdapter,
  BenchmarkCase,
  BenchmarkEnvironment,
  BenchmarkScheduleEntry,
  GenerationCallbacks,
  GenerationResult,
} from "../src/benchmark/suite/types";
import { validateGeneration } from "../src/benchmark/suite/validation";

const testCase: BenchmarkCase = {
  id: "test-32-4",
  targetInputTokens: 32,
  targetOutputTokens: 4,
  prompt: "Return alpha beta gamma delta",
  expectedPrefix: "alpha beta",
  supportsLongContext: true,
};

test("calculates interpolated percentiles and withholds p95 below 20 samples", () => {
  expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
  const small = calculateStatistics([1, 2, 3], { bootstrapResamples: 200 });
  expect(small?.p95).toBeNull();
  expect(small?.p95Status).toBe("insufficient-samples");
  const large = calculateStatistics(Array.from({ length: 20 }, (_, index) => index + 1), {
    bootstrapResamples: 200,
  });
  expect(large?.p95).toBe(19.05);
  expect(large?.p95Status).toBe("available");
});

test("bootstrap median confidence intervals are seeded and contain the median", () => {
  const values = [8, 9, 10, 10, 11, 12, 50];
  const first = bootstrapMedianConfidenceInterval(values, { seed: 123, resamples: 500 });
  const second = bootstrapMedianConfidenceInterval(values, { seed: 123, resamples: 500 });
  expect(first).toEqual(second);
  expect(first.low).toBeLessThanOrEqual(10);
  expect(first.high).toBeGreaterThanOrEqual(10);
});

test("seeded block schedule is reproducible and changes runtime order", () => {
  const options = {
    seed: 77,
    modes: ["warm-steady-state"] as const,
    workloads: [testCase],
    runtimeIds: ["owned", "transformers", "litert"],
    iterations: 8,
  };
  const first = createBenchmarkSchedule(options);
  expect(createBenchmarkSchedule(options)).toEqual(first);
  const orders = new Set(Array.from({ length: 8 }, (_, block) =>
    first.filter((entry) => entry.block === block).map((entry) => entry.runtimeId).join(",")
  ));
  expect(orders.size).toBeGreaterThan(1);
  expect(first.map((entry) => entry.sequence)).toEqual(
    Array.from({ length: first.length }, (_, index) => index),
  );
});

test("measurement ignores empty chunks and keeps multi-token chunks intact", async () => {
  const adapter = new FakeAdapter();
  const measured = await measureGeneration(adapter, testCase);
  expect(measured.external.streamChunkCount).toBe(2);
  expect(measured.chunks.map((chunk) => chunk.text)).toEqual(["alpha beta ", "gamma delta"]);
  expect(measured.result.outputTokens).toBe(4);
  expect(measured.external.streamChunkIntervalMs).toHaveLength(1);
});

test("runtime-native metrics never populate external fields", async () => {
  const measured = await measureGeneration(new FakeAdapter(), testCase);
  expect(measured.nativeMetrics).toEqual([{
    name: "decodeTokensPerSecond",
    value: 9999,
    unit: "tokens/s",
    boundary: "runtime-native",
  }]);
  expect(measured.external.aggregateDecodeTokensPerSecond).not.toBe(9999);
  expect(Object.keys(measured.external)).not.toContain("decodeTokensPerSecond");
});

test("early termination remains a correctness result but is excluded from equal work", () => {
  const validation = validateGeneration(testCase, {
    text: "alpha beta",
    stopReason: "end-token",
    inputTokens: 32,
    outputTokens: 2,
    memoryBytes: null,
  });
  expect(validation.correctness.earlyTerminated).toBe(true);
  expect(validation.equalWorkEligible).toBe(false);
  expect(validation.exclusionReasons).toContain("output-token-count-differs-materially");
});

test("runtime failures become excluded raw records instead of aborting the suite", async () => {
  const adapter = new FakeAdapter();
  adapter.failure = new Error("GPU buffer became invalid");
  const run = await runScheduledCase({
    adapter,
    testCase,
    schedule: scheduleEntry(),
    environment: fakeEnvironment("headless"),
  });
  expect(run.equalWorkEligible).toBe(false);
  expect(run.correctness.invalidOutput).toBe(true);
  expect(run.correctness.error).toContain("GPU buffer became invalid");
  expect(run.exclusionReasons).toContain("runtime-error");
});

test("runner, summary, and all report artifacts preserve separate tracks and correctness", async () => {
  const environment = fakeEnvironment("headless");
  const schedule = scheduleEntry();
  const run = await runScheduledCase({
    adapter: new FakeAdapter(),
    testCase,
    schedule,
    environment,
    startup: {
      startedAtMs: 1,
      readyAtMs: 11,
      readyMs: 10,
      bytesTransferred: 0,
      stages: [{ name: "ready", durationMs: 10, observable: true }],
      webgpuVerified: true,
      backend: "fake-webgpu",
      memoryBytes: 1024,
      notes: [],
    },
  });
  const summary = summarizeBenchmarkRuns([run], environment, [schedule], ["test limitation"]);
  expect(summary.rows.map((row) => row.track).sort()).toEqual([
    "artifact-equivalent",
    "best-available-stack",
  ]);
  const reports = renderBenchmarkArtifacts([run], summary);
  expect(JSON.parse(reports.rawResultsJsonl)).toMatchObject({ schemaVersion: 2 });
  expect(JSON.parse(reports.summaryJson)).toMatchObject({ schemaVersion: 2 });
  expect(reports.reportMarkdown).toContain("## Correctness");
  expect(reports.reportCsv).toContain("total_median_ci95_low_ms");
  expect(reports.reportHtml).toContain("Aggregate decode throughput");
});

test("summary refuses to aggregate headed and headless runs", async () => {
  const environment = fakeEnvironment("headless");
  const run = await runScheduledCase({
    adapter: new FakeAdapter(),
    testCase,
    schedule: scheduleEntry(),
    environment: fakeEnvironment("headed"),
  });
  expect(() => summarizeBenchmarkRuns([run], environment, [scheduleEntry()])).toThrow(
    "Cannot aggregate headed and headless",
  );
});

class FakeAdapter implements BenchmarkAdapter {
  readonly id = "fake-owned";
  readonly runtimeName = "Fake Owned";
  readonly runtimeVersion = "1";
  readonly modelId = "fake/gemma";
  readonly modelRevision = "abc";
  readonly artifactType = "fake safetensors";
  readonly artifactEquivalence = "pinned-source-equivalent" as const;
  readonly available = true;
  readonly limitations = [];
  failure: Error | null = null;

  async load(): Promise<never> { throw new Error("Not used"); }
  async warmup(): Promise<void> {}
  async generate(_case: BenchmarkCase, callbacks: GenerationCallbacks): Promise<GenerationResult> {
    callbacks.onRequestStart(performance.now());
    if (this.failure) throw this.failure;
    callbacks.onTextChunk("", performance.now());
    callbacks.onTextChunk("alpha beta ", performance.now());
    await Promise.resolve();
    callbacks.onTextChunk("gamma delta", performance.now());
    callbacks.onRuntimeMetric?.({
      name: "decodeTokensPerSecond",
      value: 9999,
      unit: "tokens/s",
      boundary: "runtime-native",
    });
    return {
      text: "alpha beta gamma delta",
      stopReason: "length",
      inputTokens: 32,
      outputTokens: 999,
      memoryBytes: 1024,
    };
  }
  async countTokens(text: string): Promise<number> { return text.split(/\s+/).length; }
  async resetConversation(): Promise<void> {}
  async createConversation(): Promise<void> {}
  async dispose(): Promise<void> {}
}

function scheduleEntry(): BenchmarkScheduleEntry {
  return {
    sequence: 0,
    block: 0,
    mode: "warm-steady-state",
    workloadId: testCase.id,
    iteration: 0,
    runtimeId: "fake-owned",
    conversationVariant: "not-applicable",
  };
}

function fakeEnvironment(browserMode: "headless" | "headed"): BenchmarkEnvironment {
  return {
    capturedAt: "2026-01-01T00:00:00.000Z",
    operatingSystem: "test",
    physicalDevice: "test",
    cpu: "test",
    totalRamBytes: 1,
    gpuAdapter: "fake",
    webGpuAdapterInfo: {},
    browserName: "Chrome",
    browserVersion: "1",
    browserMode,
    browserFlags: [],
    visibilityState: "visible",
    powerSource: "unknown",
    gitCommit: "test",
    benchmarkSeed: 77,
  };
}