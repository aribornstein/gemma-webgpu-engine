import { measureGeneration } from "./measurement";
import type {
  ArtifactTrack,
  BenchmarkAdapter,
  BenchmarkCase,
  BenchmarkEnvironment,
  BenchmarkScheduleEntry,
  LoadResult,
  RawBenchmarkRun,
} from "./types";
import { validateGeneration } from "./validation";

export interface RunScheduledCaseOptions {
  adapter: BenchmarkAdapter;
  testCase: BenchmarkCase;
  schedule: BenchmarkScheduleEntry;
  environment: BenchmarkEnvironment;
  runtimeLoad?: LoadResult;
  startup?: LoadResult;
}

export async function runScheduledCase(options: RunScheduledCaseOptions): Promise<RawBenchmarkRun> {
  const { adapter, testCase, schedule } = options;
  if (!adapter.available) throw new Error(`${adapter.id} is unavailable: ${adapter.limitations.join(" ")}`);
  if (schedule.mode === "conversation-cache" && schedule.conversationVariant === "fresh") {
    await adapter.resetConversation();
  }
  const startedAtMs = performance.now();
  try {
    const measured = await measureGeneration(adapter, testCase);
    const validation = validateGeneration(testCase, measured.result);
    return createRawRun(options, {
      actualInputTokens: measured.result.inputTokens,
      actualOutputTokens: measured.result.outputTokens,
      generatedCharacters: measured.result.text.length,
      outputBytes: validation.correctness.outputByteCount,
      external: measured.external,
      nativeMetrics: measured.nativeMetrics,
      correctness: validation.correctness,
      equalWorkEligible: validation.equalWorkEligible,
      exclusionReasons: validation.exclusionReasons,
      memoryBytes: measured.result.memoryBytes,
    });
  } catch (error) {
    const completionMs = performance.now();
    const message = error instanceof Error ? error.message : String(error);
    return createRawRun(options, {
      actualInputTokens: 0,
      actualOutputTokens: 0,
      generatedCharacters: 0,
      outputBytes: 0,
      external: {
        requestStartMs: startedAtMs,
        firstVisibleOutputMs: null,
        completionMs,
        ttftMs: null,
        totalMs: completionMs - startedAtMs,
        aggregateDecodeTokensPerSecond: null,
        charactersPerSecond: 0,
        streamChunkCount: 0,
        streamChunkIntervalMs: [],
      },
      nativeMetrics: [],
      correctness: {
        exactOutputText: "",
        tokenCount: 0,
        characterCount: 0,
        outputByteCount: 0,
        reachedRequestedTokenLength: false,
        matchedExpectedPrefix: false,
        invalidOutput: true,
        repeatedOutput: false,
        earlyTerminated: true,
        error: message,
      },
      equalWorkEligible: false,
      exclusionReasons: ["runtime-error", "invalid-output"],
      memoryBytes: null,
    });
  }
}

type RawRunMeasurements = Pick<
  RawBenchmarkRun,
  | "actualInputTokens"
  | "actualOutputTokens"
  | "generatedCharacters"
  | "outputBytes"
  | "external"
  | "nativeMetrics"
  | "correctness"
  | "equalWorkEligible"
  | "exclusionReasons"
  | "memoryBytes"
>;

function createRawRun(
  options: RunScheduledCaseOptions,
  measurements: RawRunMeasurements,
): RawBenchmarkRun {
  const { adapter, testCase, schedule, environment } = options;
  const trackEligibility: ArtifactTrack[] = ["best-available-stack"];
  if (adapter.artifactEquivalence !== "model-family-only") {
    trackEligibility.unshift("artifact-equivalent");
  }
  const runtimeLoad = options.runtimeLoad ?? options.startup;
  return {
    schemaVersion: 2,
    runId: `${environment.browserMode}-${schedule.sequence}-${adapter.id}-${testCase.id}`,
    capturedAt: new Date().toISOString(),
    seed: environment.benchmarkSeed,
    browserMode: environment.browserMode,
    mode: schedule.mode,
    trackEligibility: Object.freeze(trackEligibility),
    runtime: {
      id: adapter.id,
      name: adapter.runtimeName,
      version: adapter.runtimeVersion,
      modelId: adapter.modelId,
      ...(adapter.modelRevision ? { modelRevision: adapter.modelRevision } : {}),
      artifactType: adapter.artifactType,
      ...(adapter.artifactUrl ? { artifactUrl: adapter.artifactUrl } : {}),
      ...(adapter.artifactBytes ? { artifactBytes: adapter.artifactBytes } : {}),
      artifactEquivalence: adapter.artifactEquivalence,
      backend: runtimeLoad?.backend ?? "webgpu-unverified",
      webgpuVerified: runtimeLoad?.webgpuVerified ?? false,
    },
    workload: testCase,
    schedule,
    ...measurements,
    exclusionReasons: Object.freeze([...measurements.exclusionReasons]),
    ...(options.startup ? { startup: options.startup } : {}),
  };
}