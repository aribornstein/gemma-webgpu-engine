import {
  createGemmaDurableBenchmarkArtifact,
  loadGemmaDurableBenchmarkArtifact,
  saveGemmaDurableBenchmarkArtifact,
  updateGemmaDurableBenchmarkArtifact,
  type GemmaDurableBenchmarkArtifact,
  type GemmaDurableBenchmarkStorage,
} from "./durable-benchmark";
import {
  loadGemmaGenerationSession,
  type GemmaGenerationSession,
} from "./gemma-session";

export interface GemmaLongContextBenchmarkOptions {
  cacheCapacity: number;
  targetPromptTokens?: number;
  maxNewTokens?: number;
  prefillStrategy?: "auto" | "fixed-32" | "chunked-32" | "sequential";
  signal?: AbortSignal;
  storage?: GemmaDurableBenchmarkStorage;
  onArtifact?: (artifact: GemmaDurableBenchmarkArtifact) => void;
}

interface GemmaPromptCounter {
  promptTokenCount(prompt: string): number;
}

export async function runGemmaLongContextBenchmark(
  options: GemmaLongContextBenchmarkOptions,
): Promise<GemmaDurableBenchmarkArtifact> {
  const storage = options.storage ?? localStorage;
  const configuration = {
    cacheCapacity: options.cacheCapacity,
    targetPromptTokens: options.targetPromptTokens ?? options.cacheCapacity,
    maxNewTokens: options.maxNewTokens ?? 1,
    prefillStrategy: options.prefillStrategy ?? "auto",
  } as const;
  let artifact = createGemmaDurableBenchmarkArtifact(configuration);
  const checkpoint = (
    update: Parameters<typeof updateGemmaDurableBenchmarkArtifact>[2],
  ): GemmaDurableBenchmarkArtifact => {
    artifact = updateGemmaDurableBenchmarkArtifact(storage, artifact, update);
    options.onArtifact?.(artifact);
    return artifact;
  };
  saveGemmaDurableBenchmarkArtifact(storage, artifact);
  options.onArtifact?.(artifact);

  let session: GemmaGenerationSession | null = null;
  let lastPersistedPromptTokens = 0;
  try {
    throwIfAborted(options.signal);
    session = await loadGemmaGenerationSession({
      cacheCapacity: configuration.cacheCapacity,
      prefillStrategy: configuration.prefillStrategy,
    });
    throwIfAborted(options.signal);
    const memory = session.estimateRetainedGpuMemory();
    checkpoint({
      phase: "ready",
      progress: {
        promptTokens: null,
        generatedTokens: 0,
        message: "Session loaded; constructing exact prompt",
      },
      memory: {
        retainedGpuBufferCount: memory.gpuBufferCount,
        retainedGpuBufferBytes: memory.gpuBufferBytes,
      },
    });

    const prompt = createExactGemmaPrompt(session, configuration.targetPromptTokens);
    const overflow = await verifyOverflowWithSession(
      session,
      prompt,
      configuration.maxNewTokens + 1,
    );
    checkpoint({
      phase: "running",
      progress: {
        promptTokens: configuration.targetPromptTokens,
        generatedTokens: 0,
        message: "Running measured generation",
      },
      overflow,
    });
    const measured = await session.generateMeasured(prompt, {
      maxNewTokens: configuration.maxNewTokens,
      signal: options.signal,
      onPrefillProgress: (progress) => {
        const completed = progress.completedPromptTokens;
        if (completed < progress.totalPromptTokens &&
            completed - lastPersistedPromptTokens < 1_024) return;
        lastPersistedPromptTokens = completed;
        checkpoint({
          progress: {
            promptTokens: completed,
            generatedTokens: 0,
            message: `Prefill ${completed.toLocaleString()} of ${progress.totalPromptTokens.toLocaleString()} · ${progress.mode}`,
          },
        });
      },
      onToken: async ({ generatedTokenIds }) => {
        checkpoint({
          progress: {
            promptTokens: configuration.targetPromptTokens,
            generatedTokens: generatedTokenIds.length,
            message: `Generated ${generatedTokenIds.length} of ${configuration.maxNewTokens} tokens`,
          },
        });
      },
    });
    checkpoint({
      phase: "verifying",
      progress: {
        promptTokens: measured.result.promptTokenIds.length,
        generatedTokens: measured.result.generatedTokenIds.length,
        message: "Exact-fit complete; verifying wrapped prefix reuse",
      },
      result: {
        text: measured.result.text,
        generatedTokenIds: Object.freeze([...measured.result.generatedTokenIds]),
        stopReason: measured.result.stopReason,
      },
      timing: {
        ...measured.timing,
        decodeTokenMs: [...measured.timing.decodeTokenMs],
      },
      error: null,
    });
    const reused = await session.generateMeasured(prompt, {
      maxNewTokens: configuration.maxNewTokens,
      signal: options.signal,
    });
    const expectedReusedTokens = configuration.targetPromptTokens - 1;
    if (reused.timing.promptTokensReused !== expectedReusedTokens) {
      throw new Error(
        `Gemma benchmark reused ${reused.timing.promptTokensReused} prompt tokens, ` +
        `expected ${expectedReusedTokens}`,
      );
    }
    return checkpoint({
      phase: "completed",
      progress: {
        promptTokens: measured.result.promptTokenIds.length,
        generatedTokens: measured.result.generatedTokenIds.length,
        message: "Benchmark and wrapped prefix verification completed",
      },
      prefixReuse: {
        promptTokensReused: reused.timing.promptTokensReused,
        prefillMode: reused.timing.prefillMode,
        totalMs: reused.timing.totalMs,
        generatedTokenIds: Object.freeze([...reused.result.generatedTokenIds]),
        text: reused.result.text,
        stopReason: reused.result.stopReason,
      },
    });
  } catch (error) {
    const cancelled = options.signal?.aborted ?? false;
    const terminal = checkpoint({
      phase: cancelled ? "cancelled" : "failed",
      progress: {
        ...artifact.progress,
        message: cancelled ? "Benchmark cancelled" : "Benchmark failed",
      },
      error: errorMessage(error),
    });
    if (cancelled) return terminal;
    throw error;
  } finally {
    session?.destroy();
  }
}

export async function verifyGemmaLongContextOverflow(options: {
  cacheCapacity: number;
  targetPromptTokens?: number;
  maxNewTokens?: number;
}): Promise<{ requestedPositions: number; error: string }> {
  const targetPromptTokens = options.targetPromptTokens ?? options.cacheCapacity;
  const maxNewTokens = options.maxNewTokens ?? 1;
  const session = await loadGemmaGenerationSession({
    cacheCapacity: options.cacheCapacity,
    prefillStrategy: "auto",
  });
  try {
    const prompt = createExactGemmaPrompt(session, targetPromptTokens);
    return await verifyOverflowWithSession(session, prompt, maxNewTokens + 1);
  } finally {
    session.destroy();
  }
}

export function createExactGemmaPrompt(
  session: GemmaPromptCounter,
  targetPromptTokens: number,
): string {
  if (!Number.isInteger(targetPromptTokens) || targetPromptTokens < 1) {
    throw new Error("Gemma benchmark target prompt tokens must be a positive integer");
  }
  const oneUnitTokens = session.promptTokenCount(" x");
  const twoUnitTokens = session.promptTokenCount(" x x");
  const templateTokens = oneUnitTokens * 2 - twoUnitTokens;
  if (twoUnitTokens !== oneUnitTokens + 1 || targetPromptTokens < templateTokens) {
    throw new Error("Gemma benchmark cannot construct an exact synthetic prompt");
  }
  const prompt = " x".repeat(targetPromptTokens - templateTokens);
  const actualTokens = session.promptTokenCount(prompt);
  if (actualTokens !== targetPromptTokens) {
    throw new Error(
      `Gemma benchmark prompt has ${actualTokens} tokens, expected ${targetPromptTokens}`,
    );
  }
  return prompt;
}

export function recoverInterruptedGemmaBenchmark(
  storage: GemmaDurableBenchmarkStorage = localStorage,
): GemmaDurableBenchmarkArtifact | null {
  const artifact = loadGemmaDurableBenchmarkArtifact(storage);
  if (!artifact || !["loading", "ready", "running", "verifying"].includes(artifact.phase)) {
    return artifact;
  }
  return updateGemmaDurableBenchmarkArtifact(storage, artifact, {
    phase: "interrupted",
    progress: {
      ...artifact.progress,
      message: "Benchmark interrupted by page or process reload",
    },
    error: "The benchmark page or browser process reloaded before completion",
  });
}

export function serializeGemmaBenchmarkArtifact(
  artifact: GemmaDurableBenchmarkArtifact,
): string {
  return `${JSON.stringify(artifact, null, 2)}\n`;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Gemma benchmark aborted", "AbortError");
}

async function verifyOverflowWithSession(
  session: GemmaGenerationSession,
  prompt: string,
  maxNewTokens: number,
): Promise<{ requestedPositions: number; error: string }> {
  const promptTokens = session.promptTokenCount(prompt);
  const requestedPositions = promptTokens + maxNewTokens - 1;
  try {
    await session.generate(prompt, { maxNewTokens });
  } catch (error) {
    const message = errorMessage(error);
    if (!/exceeding capacity/.test(message)) throw error;
    return { requestedPositions, error: message };
  }
  throw new Error(`Gemma benchmark unexpectedly accepted ${requestedPositions} positions`);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}