import { LiteRtLmBenchmarkAdapter } from "./adapters/litert-lm";
import { OwnedWebGpuBenchmarkAdapter } from "./adapters/owned-webgpu";
import { PinnedHuggingFaceBenchmarkAdapter } from "./adapters/pinned-hugging-face";
import { TransformersJsBenchmarkAdapter } from "./adapters/transformers-js";
import { captureBrowserEnvironment, type BrowserEnvironmentInput } from "./environment";
import { runScheduledCase } from "./runner";
import type {
  BenchmarkAdapter,
  BenchmarkCase,
  BenchmarkEnvironment,
  BenchmarkScheduleEntry,
  LoadOptions,
  LoadResult,
  RawBenchmarkRun,
} from "./types";
import { calibrateWorkloadForRuntime } from "./workloads";

let adapter: BenchmarkAdapter | null = null;
let startup: LoadResult | undefined;
let environment: BenchmarkEnvironment | null = null;

export type BenchmarkRuntimeId =
  | "owned-webgpu"
  | "transformers-js"
  | "litert-lm-web"
  | "pinned-hugging-face-webgpu";

export async function initializeBenchmarkRuntime(
  runtimeId: BenchmarkRuntimeId,
  loadOptions: LoadOptions,
  environmentInput: BrowserEnvironmentInput,
): Promise<{ startup: LoadResult; environment: BenchmarkEnvironment; limitations: readonly string[] }> {
  await disposeBenchmarkRuntime();
  adapter = createAdapter(runtimeId);
  if (!adapter.available) throw new Error(`${runtimeId} is unavailable: ${adapter.limitations.join(" ")}`);
  environment = await captureBrowserEnvironment(environmentInput);
  startup = await adapter.load(loadOptions);
  return { startup, environment, limitations: adapter.limitations };
}

export async function calibrateBenchmarkWorkload(workload: BenchmarkCase): Promise<BenchmarkCase> {
  return calibrateWorkloadForRuntime(requiredAdapter(), workload);
}

export function applyStartupNetworkObservation(bytesTransferred: number, durationMs: number): void {
  if (!startup) throw new Error("Benchmark runtime startup has not completed");
  startup = {
    ...startup,
    bytesTransferred,
    stages: startup.stages.map((stage) => stage.name === "model-download"
      ? {
          name: "model-download" as const,
          durationMs,
          observable: true,
          note: "CDP-observed aggregate network activity during adapter load.",
        }
      : stage),
  };
}

export async function warmupBenchmarkRuntime(workload: BenchmarkCase, iterations: number): Promise<void> {
  const current = requiredAdapter();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    await current.warmup(workload);
  }
}

export async function prepareReusedConversation(workload: BenchmarkCase): Promise<void> {
  const current = requiredAdapter();
  await current.createConversation();
  await current.generate({ ...workload, targetOutputTokens: Math.min(16, workload.targetOutputTokens) }, {
    onRequestStart() {},
    onTextChunk() {},
  });
}

export async function executeBenchmarkScheduleEntry(
  workload: BenchmarkCase,
  schedule: BenchmarkScheduleEntry,
): Promise<RawBenchmarkRun> {
  if (!environment) throw new Error("Benchmark environment has not been captured");
  return runScheduledCase({
    adapter: requiredAdapter(),
    testCase: workload,
    schedule,
    environment,
    ...(startup ? { runtimeLoad: startup } : {}),
    ...(startup && (schedule.mode === "network-cold-startup" || schedule.mode === "cached-cold-startup")
      ? { startup }
      : {}),
  });
}

export async function disposeBenchmarkRuntime(): Promise<void> {
  if (adapter) await adapter.dispose();
  adapter = null;
  startup = undefined;
  environment = null;
}

function requiredAdapter(): BenchmarkAdapter {
  if (!adapter) throw new Error("Benchmark runtime has not been initialized");
  return adapter;
}

function createAdapter(runtimeId: BenchmarkRuntimeId): BenchmarkAdapter {
  switch (runtimeId) {
    case "owned-webgpu": return new OwnedWebGpuBenchmarkAdapter();
    case "transformers-js": return new TransformersJsBenchmarkAdapter();
    case "litert-lm-web": return new LiteRtLmBenchmarkAdapter();
    case "pinned-hugging-face-webgpu": return new PinnedHuggingFaceBenchmarkAdapter();
  }
}