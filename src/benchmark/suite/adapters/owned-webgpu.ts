import { GEMMA_4_E2B_CACHE_SPEC } from "../../../model/cached-safetensors";
import {
  loadGemmaGenerationSession,
  type GemmaGenerationSession,
} from "../../../runtime/gemma-session";
import { loadGemmaTokenizer, type GemmaTokenizer } from "../../../runtime/gemma-tokenizer";
import type {
  BenchmarkAdapter,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
} from "../types";

const MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers";
const MODEL_REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";

export class OwnedWebGpuBenchmarkAdapter implements BenchmarkAdapter {
  readonly id = "owned-webgpu";
  readonly runtimeName = "Owned WebGPU";
  readonly runtimeVersion = "workspace";
  readonly modelId = MODEL_ID;
  readonly modelRevision = MODEL_REVISION;
  readonly artifactType = "mobile-QAT safetensors";
  readonly artifactUrl = GEMMA_4_E2B_CACHE_SPEC.sourceKey;
  readonly artifactBytes = GEMMA_4_E2B_CACHE_SPEC.fileSize;
  readonly artifactEquivalence = "pinned-source-equivalent" as const;
  readonly available = true;
  readonly limitations = Object.freeze([
    "The public session loader does not separately expose parsing, GPU upload, or shader compilation durations.",
    "EOS suppression is not supported by the owned greedy path; early termination is recorded and excluded from equal-work throughput.",
  ]);

  private session: GemmaGenerationSession | null = null;
  private tokenizer: GemmaTokenizer | null = null;
  private reusePromptCache = false;

  async load(options: LoadOptions): Promise<LoadResult> {
    await this.dispose();
    const startedAtMs = performance.now();
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Owned benchmark requires a WebGPU adapter");
    [this.session, this.tokenizer] = await Promise.all([
      loadGemmaGenerationSession({
        cacheCapacity: options.cacheCapacity,
        sourceUrl: options.sourceUrl,
        prefillStrategy: "auto",
      }),
      loadGemmaTokenizer(),
    ]);
    const readyAtMs = performance.now();
    const memory = this.session.estimateRetainedGpuMemory();
    return {
      startedAtMs,
      readyAtMs,
      readyMs: readyAtMs - startedAtMs,
      bytesTransferred: 0,
      stages: [
        unobservable("model-download", "Measured by the browser network observer."),
        unobservable("cached-asset-read", "Included in aggregate ready time."),
        unobservable("parse-deserialize", "Not exposed by the owned loader."),
        unobservable("graph-create", "Included in aggregate ready time."),
        unobservable("gpu-upload", "Included in aggregate ready time."),
        unobservable("shader-compile", "Pipeline creation is included in aggregate ready time."),
        { name: "ready", durationMs: readyAtMs - startedAtMs, observable: true },
      ],
      webgpuVerified: adapter.features.has("subgroups") && adapter.features.has("shader-f16"),
      backend: "webgpu",
      memoryBytes: memory.gpuBufferBytes,
      notes: this.limitations,
    };
  }

  async warmup(testCase: BenchmarkCase): Promise<void> {
    await this.generate(testCase, emptyCallbacks());
  }

  async generate(
    testCase: BenchmarkCase,
    callbacks: GenerationCallbacks,
  ): Promise<GenerationResult> {
    const session = required(this.session, "Owned benchmark adapter is not loaded");
    let previousText = "";
    callbacks.onRequestStart(performance.now());
    const measured = await session.generateMeasured(testCase.prompt, {
      maxNewTokens: testCase.targetOutputTokens,
      reusePromptCache: this.reusePromptCache,
      onToken(update) {
        const text = update.text;
        const chunk = text.startsWith(previousText) ? text.slice(previousText.length) : text;
        previousText = text;
        callbacks.onTextChunk(chunk, performance.now());
      },
    });
    callbacks.onRuntimeMetric?.({
      name: "prefillMs",
      value: measured.timing.prefillMs,
      unit: "ms",
      boundary: "runtime-native",
    });
    callbacks.onRuntimeMetric?.({
      name: "promptTokensReused",
      value: measured.timing.promptTokensReused,
      unit: "tokens",
      boundary: "runtime-native",
    });
    const memory = session.estimateRetainedGpuMemory();
    return {
      text: measured.result.text,
      stopReason: measured.result.stopReason,
      inputTokens: measured.result.promptTokenIds.length,
      outputTokens: measured.result.generatedTokenIds.length,
      outputTokenIds: Object.freeze([...measured.result.generatedTokenIds]),
      memoryBytes: memory.gpuBufferBytes,
    };
  }

  async countTokens(text: string, purpose: "input" | "output" = "input"): Promise<number> {
    const tokenizer = required(this.tokenizer, "Owned benchmark tokenizer is not loaded");
    return (purpose === "input" ? tokenizer.encodePrompt(text) : tokenizer.encodeText(text)).length;
  }

  async resetConversation(): Promise<void> {
    this.reusePromptCache = false;
  }

  async createConversation(): Promise<void> {
    this.reusePromptCache = true;
  }

  async dispose(): Promise<void> {
    this.session?.destroy();
    this.session = null;
    this.tokenizer = null;
    this.reusePromptCache = false;
  }
}

function unobservable(name: Parameters<typeof stage>[0], note: string) {
  return stage(name, null, false, note);
}

function stage(
  name: Exclude<LoadResult["stages"][number]["name"], "page-startup" | "runtime-create">,
  durationMs: number | null,
  observable: boolean,
  note?: string,
): LoadResult["stages"][number] {
  return { name, durationMs, observable, ...(note ? { note } : {}) };
}

function required<T>(value: T | null, message: string): T {
  if (value === null) throw new Error(message);
  return value;
}

function emptyCallbacks(): GenerationCallbacks {
  return { onRequestStart() {}, onTextChunk() {} };
}