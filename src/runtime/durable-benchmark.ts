export const GEMMA_DURABLE_BENCHMARK_STORAGE_KEY = "gemma-webgpu-engine:long-context-benchmark:v1";

export type GemmaDurableBenchmarkPhase =
  | "loading"
  | "ready"
  | "running"
  | "verifying"
  | "completed"
  | "cancelled"
  | "failed"
  | "interrupted";

export interface GemmaDurableBenchmarkArtifact {
  schemaVersion: 1;
  runId: string;
  createdAt: string;
  updatedAt: string;
  phase: GemmaDurableBenchmarkPhase;
  configuration: {
    cacheCapacity: number;
    targetPromptTokens: number;
    maxNewTokens: number;
    prefillStrategy: "auto" | "fixed-32" | "chunked-32" | "sequential";
  };
  progress: {
    promptTokens: number | null;
    generatedTokens: number;
    message: string;
  };
  memory: {
    retainedGpuBufferCount: number;
    retainedGpuBufferBytes: number;
  } | null;
  result: {
    text: string;
    generatedTokenIds: readonly number[];
    stopReason: string;
  } | null;
  timing: Record<string, unknown> | null;
  error: string | null;
  prefixReuse: {
    promptTokensReused: number;
    prefillMode: string;
    totalMs: number;
    generatedTokenIds: readonly number[];
    text: string;
    stopReason: string;
  } | null;
  overflow: {
    requestedPositions: number;
    error: string;
  } | null;
}

export interface GemmaDurableBenchmarkStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export function createGemmaDurableBenchmarkArtifact(
  configuration: GemmaDurableBenchmarkArtifact["configuration"],
  runId = crypto.randomUUID(),
  now = new Date(),
): GemmaDurableBenchmarkArtifact {
  validateConfiguration(configuration);
  const timestamp = now.toISOString();
  return {
    schemaVersion: 1,
    runId,
    createdAt: timestamp,
    updatedAt: timestamp,
    phase: "loading",
    configuration: { ...configuration },
    progress: {
      promptTokens: null,
      generatedTokens: 0,
      message: "Loading benchmark session",
    },
    memory: null,
    result: null,
    timing: null,
    error: null,
    prefixReuse: null,
    overflow: null,
  };
}

export function saveGemmaDurableBenchmarkArtifact(
  storage: GemmaDurableBenchmarkStorage,
  artifact: GemmaDurableBenchmarkArtifact,
): void {
  validateArtifact(artifact);
  storage.setItem(GEMMA_DURABLE_BENCHMARK_STORAGE_KEY, JSON.stringify(artifact));
}

export function loadGemmaDurableBenchmarkArtifact(
  storage: GemmaDurableBenchmarkStorage,
): GemmaDurableBenchmarkArtifact | null {
  const stored = storage.getItem(GEMMA_DURABLE_BENCHMARK_STORAGE_KEY);
  if (stored === null) return null;
  const artifact: unknown = JSON.parse(stored);
  validateArtifact(artifact);
  return artifact;
}

export function updateGemmaDurableBenchmarkArtifact(
  storage: GemmaDurableBenchmarkStorage,
  artifact: GemmaDurableBenchmarkArtifact,
  update: Partial<Omit<GemmaDurableBenchmarkArtifact, "schemaVersion" | "runId" | "createdAt" | "configuration">>,
  now = new Date(),
): GemmaDurableBenchmarkArtifact {
  const next: GemmaDurableBenchmarkArtifact = {
    ...artifact,
    ...update,
    updatedAt: now.toISOString(),
  };
  saveGemmaDurableBenchmarkArtifact(storage, next);
  return next;
}

export function clearGemmaDurableBenchmarkArtifact(
  storage: GemmaDurableBenchmarkStorage,
): void {
  storage.removeItem(GEMMA_DURABLE_BENCHMARK_STORAGE_KEY);
}

function validateConfiguration(
  configuration: GemmaDurableBenchmarkArtifact["configuration"],
): void {
  for (const [name, value] of Object.entries({
    cacheCapacity: configuration.cacheCapacity,
    targetPromptTokens: configuration.targetPromptTokens,
    maxNewTokens: configuration.maxNewTokens,
  })) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`Gemma durable benchmark ${name} must be a positive integer`);
    }
  }
  if (configuration.targetPromptTokens + configuration.maxNewTokens - 1 >
      configuration.cacheCapacity) {
    throw new Error("Gemma durable benchmark prompt and output exceed cache capacity");
  }
}

function validateArtifact(value: unknown): asserts value is GemmaDurableBenchmarkArtifact {
  if (!value || typeof value !== "object") {
    throw new Error("Gemma durable benchmark artifact must be an object");
  }
  const artifact = value as Partial<GemmaDurableBenchmarkArtifact>;
  if (artifact.schemaVersion !== 1 || typeof artifact.runId !== "string" ||
      typeof artifact.createdAt !== "string" || typeof artifact.updatedAt !== "string" ||
      typeof artifact.phase !== "string" || !artifact.configuration || !artifact.progress) {
    throw new Error("Gemma durable benchmark artifact is invalid");
  }
  validateConfiguration(artifact.configuration);
}