export type BenchmarkMode =
  | "network-cold-startup"
  | "cached-cold-startup"
  | "warm-steady-state"
  | "conversation-cache";

export type BrowserMode = "headless" | "headed";
export type ArtifactTrack = "artifact-equivalent" | "best-available-stack";
export type ArtifactEquivalence = "pinned-source-equivalent" | "demonstrated-equivalent" | "model-family-only";

export interface BenchmarkCase {
  id: string;
  targetInputTokens: number;
  targetOutputTokens: number;
  prompt: string;
  expectedPrefix: string;
  supportsLongContext: boolean;
}

export interface LoadOptions {
  mode: Extract<BenchmarkMode, "network-cold-startup" | "cached-cold-startup">;
  cacheCapacity: number;
  sourceUrl?: string;
}

export interface StartupStage {
  name:
    | "page-startup"
    | "model-download"
    | "cached-asset-read"
    | "parse-deserialize"
    | "runtime-create"
    | "graph-create"
    | "gpu-upload"
    | "shader-compile"
    | "ready";
  durationMs: number | null;
  observable: boolean;
  note?: string;
}

export interface LoadResult {
  startedAtMs: number;
  readyAtMs: number;
  readyMs: number;
  bytesTransferred: number;
  stages: readonly StartupStage[];
  webgpuVerified: boolean;
  backend: string;
  memoryBytes: number | null;
  notes: readonly string[];
}

export interface RuntimeMetric {
  name: string;
  value: number;
  unit: string;
  boundary: "runtime-native";
}

export interface GenerationCallbacks {
  onRequestStart(timestampMs: number): void;
  onTextChunk(text: string, timestampMs: number): void;
  onRuntimeMetric?(metric: RuntimeMetric): void;
}

export interface GenerationResult {
  text: string;
  stopReason: "length" | "end-token" | "stop-token" | "error" | "unknown";
  inputTokens: number;
  outputTokens: number;
  outputTokenIds?: readonly number[];
  memoryBytes: number | null;
  error?: string;
}

export interface BenchmarkAdapter {
  readonly id: string;
  readonly runtimeName: string;
  readonly runtimeVersion: string;
  readonly modelId: string;
  readonly modelRevision?: string;
  readonly artifactType: string;
  readonly artifactUrl?: string;
  readonly artifactBytes?: number;
  readonly artifactEquivalence: ArtifactEquivalence;
  readonly available: boolean;
  readonly limitations: readonly string[];

  load(options: LoadOptions): Promise<LoadResult>;
  warmup(testCase: BenchmarkCase): Promise<void>;
  generate(testCase: BenchmarkCase, callbacks: GenerationCallbacks): Promise<GenerationResult>;
  countTokens(text: string, purpose?: "input" | "output"): Promise<number>;
  resetConversation(): Promise<void>;
  createConversation(): Promise<void>;
  dispose(): Promise<void>;
}

export interface BenchmarkEnvironment {
  capturedAt: string;
  operatingSystem: string;
  physicalDevice: string;
  cpu: string;
  totalRamBytes: number;
  gpuAdapter: string;
  webGpuAdapterInfo: Record<string, string | number | boolean>;
  browserName: string;
  browserVersion: string;
  browserMode: BrowserMode;
  browserFlags: readonly string[];
  visibilityState: DocumentVisibilityState;
  powerSource: "battery" | "external-power" | "unknown";
  gitCommit: string;
  benchmarkSeed: number;
}

export interface BenchmarkScheduleEntry {
  sequence: number;
  block: number;
  mode: BenchmarkMode;
  workloadId: string;
  iteration: number;
  runtimeId: string;
  conversationVariant: "reused" | "fresh" | "not-applicable";
}

export interface ExternalGenerationMetrics {
  requestStartMs: number;
  firstVisibleOutputMs: number | null;
  completionMs: number;
  ttftMs: number | null;
  totalMs: number;
  aggregateDecodeTokensPerSecond: number | null;
  charactersPerSecond: number;
  streamChunkCount: number;
  streamChunkIntervalMs: readonly number[];
}

export interface CorrectnessResult {
  exactOutputText: string;
  tokenCount: number;
  characterCount: number;
  outputByteCount: number;
  reachedRequestedTokenLength: boolean;
  matchedExpectedPrefix: boolean;
  invalidOutput: boolean;
  repeatedOutput: boolean;
  earlyTerminated: boolean;
  error: string | null;
}

export interface RawBenchmarkRun {
  schemaVersion: 2;
  runId: string;
  capturedAt: string;
  seed: number;
  browserMode: BrowserMode;
  mode: BenchmarkMode;
  trackEligibility: readonly ArtifactTrack[];
  runtime: {
    id: string;
    name: string;
    version: string;
    modelId: string;
    modelRevision?: string;
    artifactType: string;
    artifactUrl?: string;
    artifactBytes?: number;
    artifactEquivalence: ArtifactEquivalence;
    backend: string;
    webgpuVerified: boolean;
  };
  workload: BenchmarkCase;
  schedule: BenchmarkScheduleEntry;
  actualInputTokens: number;
  actualOutputTokens: number;
  generatedCharacters: number;
  outputBytes: number;
  external: ExternalGenerationMetrics;
  nativeMetrics: readonly RuntimeMetric[];
  correctness: CorrectnessResult;
  equalWorkEligible: boolean;
  exclusionReasons: readonly string[];
  memoryBytes: number | null;
  startup?: LoadResult;
}

export interface MetricStatistics {
  sampleCount: number;
  median: number;
  mean: number;
  standardDeviation: number;
  minimum: number;
  maximum: number;
  p50: number;
  p90: number;
  p95: number | null;
  p95Status: "available" | "insufficient-samples";
  coefficientOfVariation: number | null;
  medianConfidenceInterval95: { low: number; high: number };
}

export interface SummaryRow {
  track: ArtifactTrack;
  browserMode: BrowserMode;
  mode: BenchmarkMode;
  conversationVariant: BenchmarkScheduleEntry["conversationVariant"];
  runtimeId: string;
  runtimeName: string;
  workloadId: string;
  targetInputTokens: number;
  targetOutputTokens: number;
  validRuns: number;
  excludedRuns: number;
  actualInputTokens: MetricStatistics | null;
  actualOutputTokens: MetricStatistics | null;
  ttftMs: MetricStatistics | null;
  totalMs: MetricStatistics | null;
  aggregateDecodeTokensPerSecond: MetricStatistics | null;
  charactersPerSecond: MetricStatistics | null;
  streamChunkIntervalMs: MetricStatistics | null;
  startupReadyMs: MetricStatistics | null;
}

export interface BenchmarkSummary {
  schemaVersion: 2;
  generatedAt: string;
  environment: BenchmarkEnvironment;
  schedule: readonly BenchmarkScheduleEntry[];
  rows: readonly SummaryRow[];
  limitations: readonly string[];
}