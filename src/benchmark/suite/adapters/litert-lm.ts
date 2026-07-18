import { Engine, SamplerType, type Conversation, type Message } from "@litert-lm/core";
import type {
  BenchmarkAdapter,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
} from "../types";

const MODEL_URL =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm";

export class LiteRtLmBenchmarkAdapter implements BenchmarkAdapter {
  readonly id = "litert-lm-web";
  readonly runtimeName = "LiteRT-LM Web";
  readonly runtimeVersion = "0.14.0";
  readonly modelId = "litert-community/gemma-4-E2B-it-litert-lm";
  readonly artifactType = "LiteRT-LM Web bundle";
  readonly artifactUrl = MODEL_URL;
  readonly artifactBytes = 2_008_000_000;
  readonly artifactEquivalence = "model-family-only" as const;
  readonly available = true;
  readonly limitations = Object.freeze([
    "Streaming callbacks may contain multiple tokens; only stream chunk intervals are reported externally.",
    "The public API does not expose an arbitrary model-tokenizer encode operation; native prefill/decode counts are retained separately.",
    "EOS suppression is not exposed by the preview API; early termination is recorded and excluded from equal-work throughput.",
  ]);

  private engine: Engine | null = null;
  private conversation: Conversation | null = null;
  private lastOutput: { text: string; tokens: number } | null = null;

  async load(options: LoadOptions): Promise<LoadResult> {
    await this.dispose();
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("LiteRT-LM benchmark requires WebGPU");
    const startedAtMs = performance.now();
    this.engine = await Engine.create({
      model: MODEL_URL,
      mainExecutorSettings: { maxNumTokens: options.cacheCapacity },
      benchmarkEnabled: true,
    });
    const readyAtMs = performance.now();
    return {
      startedAtMs,
      readyAtMs,
      readyMs: readyAtMs - startedAtMs,
      bytesTransferred: 0,
      stages: [
        unavailableStage("model-download", "Measured by the browser network observer."),
        unavailableStage("cached-asset-read", "Included in aggregate ready time."),
        unavailableStage("parse-deserialize", "Not separately exposed."),
        unavailableStage("runtime-create", "Engine.create combines runtime and model setup."),
        unavailableStage("graph-create", "Not separately exposed."),
        unavailableStage("gpu-upload", "Not separately exposed."),
        unavailableStage("shader-compile", "Not separately exposed."),
        { name: "ready", durationMs: readyAtMs - startedAtMs, observable: true },
      ],
      webgpuVerified: true,
      backend: "webgpu/litert-lm-wasm",
      memoryBytes: null,
      notes: this.limitations,
    };
  }

  async warmup(testCase: BenchmarkCase): Promise<void> {
    await this.resetConversation();
    await this.generate(testCase, emptyCallbacks());
  }

  async generate(testCase: BenchmarkCase, callbacks: GenerationCallbacks): Promise<GenerationResult> {
    const engine = required(this.engine, "LiteRT-LM adapter is not loaded");
    const conversation = this.conversation ?? await engine.createConversation({
      sessionConfig: {
        maxOutputTokens: testCase.targetOutputTokens,
        samplerParams: { type: SamplerType.GREEDY, temperature: 0, seed: 42 },
      },
    });
    const deleteAfterRun = this.conversation === null;
    let text = "";
    try {
      callbacks.onRequestStart(performance.now());
      for await (const message of conversation.sendMessageStreaming({
        role: "user",
        content: testCase.prompt,
      })) {
        const next = messageText(message);
        const chunk = streamDelta(text, next);
        text = mergeStreamText(text, next);
        callbacks.onTextChunk(chunk, performance.now());
      }
      const native = await conversation.getBenchmarkInfo();
      callbacks.onRuntimeMetric?.({ name: "prefillTokensPerSecond", value: native.lastPrefillTokensPerSecond, unit: "tokens/s", boundary: "runtime-native" });
      callbacks.onRuntimeMetric?.({ name: "prefillTokenCount", value: native.lastPrefillTokenCount, unit: "tokens", boundary: "runtime-native" });
      callbacks.onRuntimeMetric?.({ name: "decodeTokensPerSecond", value: native.lastDecodeTokensPerSecond, unit: "tokens/s", boundary: "runtime-native" });
      callbacks.onRuntimeMetric?.({ name: "timeToFirstTokenMs", value: native.timeToFirstTokenInSecond * 1000, unit: "ms", boundary: "runtime-native" });
      this.lastOutput = { text, tokens: native.lastDecodeTokenCount };
      return {
        text,
        stopReason: native.lastDecodeTokenCount >= testCase.targetOutputTokens ? "length" : "unknown",
        inputTokens: native.lastPrefillTokenCount,
        outputTokens: native.lastDecodeTokenCount,
        memoryBytes: null,
      };
    } finally {
      if (deleteAfterRun) await conversation.delete();
    }
  }

  async countTokens(text: string, purpose: "input" | "output" = "input"): Promise<number> {
    if (purpose === "output" && this.lastOutput?.text === text) return this.lastOutput.tokens;
    throw new Error(`LiteRT-LM does not expose arbitrary ${purpose} tokenizer encoding`);
  }

  async resetConversation(): Promise<void> {
    if (this.conversation) await this.conversation.delete();
    this.conversation = null;
  }

  async createConversation(): Promise<void> {
    await this.resetConversation();
    const engine = required(this.engine, "LiteRT-LM adapter is not loaded");
    this.conversation = await engine.createConversation({
      sessionConfig: {
        maxOutputTokens: 512,
        samplerParams: { type: SamplerType.GREEDY, temperature: 0, seed: 42 },
      },
    });
  }

  async dispose(): Promise<void> {
    await this.resetConversation();
    if (this.engine) await this.engine.delete();
    this.engine = null;
    this.lastOutput = null;
  }
}

function unavailableStage(name: LoadResult["stages"][number]["name"], note: string) {
  return { name, durationMs: null, observable: false, note } as const;
}

function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  return (message.content ?? []).flatMap((part) => part.type === "text" ? [part.text] : []).join("");
}

function mergeStreamText(accumulated: string, next: string): string {
  if (next.startsWith(accumulated)) return next;
  if (accumulated.endsWith(next)) return accumulated;
  return accumulated + next;
}

function streamDelta(accumulated: string, next: string): string {
  return next.startsWith(accumulated) ? next.slice(accumulated.length) : next;
}

function required<T>(value: T | null, message: string): T {
  if (value === null) throw new Error(message);
  return value;
}

function emptyCallbacks(): GenerationCallbacks {
  return { onRequestStart() {}, onTextChunk() {} };
}