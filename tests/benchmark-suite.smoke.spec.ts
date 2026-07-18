import { expect, test } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { renderBenchmarkArtifacts } from "../src/benchmark/suite/report";
import { runScheduledCase } from "../src/benchmark/suite/runner";
import { createBenchmarkSchedule } from "../src/benchmark/suite/schedule";
import { summarizeBenchmarkRuns } from "../src/benchmark/suite/summary";
import type {
  BenchmarkAdapter,
  BenchmarkEnvironment,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
  RawBenchmarkRun,
} from "../src/benchmark/suite/types";
import { createSmokeWorkload } from "../src/benchmark/suite/workloads";

const ENABLED = process.env.BENCHMARK_SUITE_SYNTHETIC_SMOKE === "1";

test.skip(!ENABLED, "Set BENCHMARK_SUITE_SYNTHETIC_SMOKE=1 to write smoke artifacts");

test("writes all rigorous suite artifacts through a synthetic adapter", async ({ browser }) => {
  const seed = 20260717;
  const workload = createSmokeWorkload();
  const adapter = new SyntheticSmokeAdapter();
  const environment: BenchmarkEnvironment = {
    capturedAt: new Date().toISOString(),
    operatingSystem: `${os.type()} ${os.release()} ${os.arch()}`,
    physicalDevice: os.hostname(),
    cpu: os.cpus()[0]?.model ?? "unknown",
    totalRamBytes: os.totalmem(),
    gpuAdapter: "synthetic-smoke-no-hardware-result",
    webGpuAdapterInfo: {},
    browserName: "Chrome",
    browserVersion: browser.version(),
    browserMode: "headless",
    browserFlags: [],
    visibilityState: "visible",
    powerSource: "unknown",
    gitCommit: "synthetic-smoke",
    benchmarkSeed: seed,
  };
  const schedule = createBenchmarkSchedule({
    seed,
    modes: [
      "network-cold-startup",
      "cached-cold-startup",
      "warm-steady-state",
      "conversation-cache",
    ],
    workloads: [workload],
    runtimeIds: [adapter.id],
    iterations: 1,
  });
  const startup = await adapter.load({ mode: "network-cold-startup", cacheCapacity: 64 });
  await adapter.warmup(workload);
  const runs: RawBenchmarkRun[] = [];
  for (const entry of schedule) {
    if (entry.conversationVariant === "reused") await adapter.createConversation();
    runs.push(await runScheduledCase({
      adapter,
      testCase: workload,
      schedule: entry,
      environment,
      ...(entry.mode.endsWith("startup") ? { startup } : {}),
    }));
  }
  await adapter.dispose();
  const summary = summarizeBenchmarkRuns(runs, environment, schedule, [
    "Synthetic smoke data validates orchestration and report structure only; it is not performance evidence.",
  ]);
  const reports = renderBenchmarkArtifacts(runs, summary);
  const outputDirectory = path.join(process.cwd(), "benchmarks", "suite-smoke", "headless");
  await mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    writeFile(path.join(outputDirectory, "raw-results.jsonl"), reports.rawResultsJsonl),
    writeFile(path.join(outputDirectory, "summary.json"), reports.summaryJson),
    writeFile(path.join(outputDirectory, "report.md"), reports.reportMarkdown),
    writeFile(path.join(outputDirectory, "report.csv"), reports.reportCsv),
    writeFile(path.join(outputDirectory, "report.html"), reports.reportHtml),
  ]);
  expect(runs).toHaveLength(5);
  expect(summary.rows).toHaveLength(10);
});

class SyntheticSmokeAdapter implements BenchmarkAdapter {
  readonly id = "synthetic-smoke";
  readonly runtimeName = "Synthetic Smoke Adapter";
  readonly runtimeVersion = "1";
  readonly modelId = "synthetic/gemma-4-e2b";
  readonly modelRevision = "not-a-model";
  readonly artifactType = "synthetic";
  readonly artifactEquivalence = "demonstrated-equivalent" as const;
  readonly available = true;
  readonly limitations = ["No model or GPU is used by this smoke adapter."];

  async load(_options: LoadOptions): Promise<LoadResult> {
    const startedAtMs = performance.now();
    return {
      startedAtMs,
      readyAtMs: startedAtMs + 1,
      readyMs: 1,
      bytesTransferred: 128,
      stages: [{ name: "ready", durationMs: 1, observable: true }],
      webgpuVerified: false,
      backend: "synthetic",
      memoryBytes: 0,
      notes: this.limitations,
    };
  }
  async warmup(_testCase: BenchmarkCase): Promise<void> {}
  async generate(_testCase: unknown, callbacks: GenerationCallbacks): Promise<GenerationResult> {
    callbacks.onRequestStart(performance.now());
    callbacks.onTextChunk("alpha beta gamma delta ", performance.now());
    callbacks.onTextChunk("alpha beta gamma delta", performance.now());
    callbacks.onRuntimeMetric?.({
      name: "syntheticNativeLatency",
      value: 0.1,
      unit: "ms",
      boundary: "runtime-native",
    });
    return {
      text: "alpha beta gamma delta alpha beta gamma delta",
      stopReason: "length",
      inputTokens: 32,
      outputTokens: 8,
      memoryBytes: 0,
    };
  }
  async countTokens(text: string): Promise<number> { return text.trim().split(/\s+/).length; }
  async resetConversation(): Promise<void> {}
  async createConversation(): Promise<void> {}
  async dispose(): Promise<void> {}
}