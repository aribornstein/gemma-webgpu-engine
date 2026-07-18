import {
  AutoProcessor,
  Gemma4ForCausalLM,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import type {
  BenchmarkAdapter,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
} from "../types";

const MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const MODEL_REVISION = "9f4bef82ea6e296bc69f8a2f5939f73af81b07a6";

export class TransformersJsBenchmarkAdapter implements BenchmarkAdapter {
  readonly id = "transformers-js";
  readonly runtimeName = "Transformers.js";
  readonly runtimeVersion = "4.2.0";
  readonly modelId = MODEL_ID;
  readonly modelRevision = MODEL_REVISION;
  readonly artifactType = "ONNX q4f16 text-only export";
  readonly artifactUrl = `https://huggingface.co/${MODEL_ID}/tree/${MODEL_REVISION}`;
  readonly artifactBytes = 3_111_069_636;
  readonly artifactEquivalence = "model-family-only" as const;
  readonly available = true;
  readonly limitations = Object.freeze([
    "TextStreamer callbacks are JavaScript-visible token/text events, not GPU completion timestamps.",
    "The loader does not expose parse, graph creation, GPU upload, and shader compilation as separate stages.",
  ]);

  private processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null = null;
  private model: Awaited<ReturnType<typeof Gemma4ForCausalLM.from_pretrained>> | null = null;
  private messages: Array<{ role: "user" | "assistant"; content: Array<{ type: "text"; text: string }> }> = [];
  private conversationEnabled = false;
  private previousEnvironment: { allowLocalModels: boolean; allowRemoteModels: boolean; useBrowserCache: boolean } | null = null;

  async load(_options: LoadOptions): Promise<LoadResult> {
    await this.dispose();
    this.previousEnvironment = {
      allowLocalModels: env.allowLocalModels,
      allowRemoteModels: env.allowRemoteModels,
      useBrowserCache: env.useBrowserCache,
    };
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Transformers.js benchmark requires WebGPU");
    const startedAtMs = performance.now();
    [this.processor, this.model] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID, { revision: MODEL_REVISION }),
      Gemma4ForCausalLM.from_pretrained(MODEL_ID, {
        revision: MODEL_REVISION,
        device: "webgpu",
        dtype: "q4f16",
      }),
    ]);
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
        unavailableStage("graph-create", "Not separately exposed."),
        unavailableStage("gpu-upload", "Not separately exposed."),
        unavailableStage("shader-compile", "Not separately exposed."),
        { name: "ready", durationMs: readyAtMs - startedAtMs, observable: true },
      ],
      webgpuVerified: true,
      backend: "webgpu/onnxruntime-web",
      memoryBytes: null,
      notes: this.limitations,
    };
  }

  async warmup(testCase: BenchmarkCase): Promise<void> {
    await this.generate(testCase, emptyCallbacks());
  }

  async generate(testCase: BenchmarkCase, callbacks: GenerationCallbacks): Promise<GenerationResult> {
    const processor = required(this.processor, "Transformers.js processor is not loaded");
    const model = required(this.model, "Transformers.js model is not loaded");
    const messages = this.conversationEnabled
      ? [...this.messages, userMessage(testCase.prompt)]
      : [userMessage(testCase.prompt)];
    const formatted = processor.apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    });
    if (typeof formatted !== "string") throw new Error("Chat template did not return text");
    const inputs = await processor(formatted, null, null, { add_special_tokens: false });
    const inputTokens = inputs.input_ids.dims.at(-1) ?? 0;
    const generatedTokenIds: number[] = [];
    let streamedText = "";
    const streamer = new TextStreamer(processor.tokenizer!, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function(text) {
        streamedText += text;
        callbacks.onTextChunk(text, performance.now());
      },
      token_callback_function(tokens) {
        generatedTokenIds.push(...tokens.map(Number));
      },
    });
    callbacks.onRequestStart(performance.now());
    try {
      const output = await model.generate({
        ...inputs,
        min_new_tokens: testCase.targetOutputTokens,
        max_new_tokens: testCase.targetOutputTokens,
        do_sample: false,
        streamer,
      });
      disposeTensorValues(output);
    } finally {
      disposeTensorValues(inputs);
    }
    const endIds = eosIds(model);
    const outputTokenIds = generatedTokenIds.filter((tokenId) => !endIds.has(tokenId));
    const text = streamedText || processor.decode(outputTokenIds, { skip_special_tokens: true });
    if (this.conversationEnabled) {
      this.messages.push(userMessage(testCase.prompt), assistantMessage(text));
    }
    return {
      text,
      stopReason: outputTokenIds.length >= testCase.targetOutputTokens ? "length" : "end-token",
      inputTokens,
      outputTokens: outputTokenIds.length,
      outputTokenIds: Object.freeze(outputTokenIds),
      memoryBytes: null,
    };
  }

  async countTokens(text: string, purpose: "input" | "output" = "input"): Promise<number> {
    const processor = required(this.processor, "Transformers.js processor is not loaded");
    let input = text;
    if (purpose === "input") {
      const formatted = processor.apply_chat_template([userMessage(text)], {
        add_generation_prompt: true,
        tokenize: false,
      });
      if (typeof formatted !== "string") throw new Error("Chat template did not return text");
      input = formatted;
    }
    const encoded = await processor(input, null, null, { add_special_tokens: false });
    try {
      return encoded.input_ids.dims.at(-1) ?? 0;
    } finally {
      disposeTensorValues(encoded);
    }
  }

  async resetConversation(): Promise<void> {
    this.messages = [];
    this.conversationEnabled = false;
  }

  async createConversation(): Promise<void> {
    this.messages = [];
    this.conversationEnabled = true;
  }

  async dispose(): Promise<void> {
    if (this.model) await this.model.dispose();
    this.model = null;
    this.processor = null;
    this.messages = [];
    this.conversationEnabled = false;
    if (this.previousEnvironment) {
      env.allowLocalModels = this.previousEnvironment.allowLocalModels;
      env.allowRemoteModels = this.previousEnvironment.allowRemoteModels;
      env.useBrowserCache = this.previousEnvironment.useBrowserCache;
      this.previousEnvironment = null;
    }
  }
}

function userMessage(text: string) {
  return { role: "user" as const, content: [{ type: "text" as const, text }] };
}

function assistantMessage(text: string) {
  return { role: "assistant" as const, content: [{ type: "text" as const, text }] };
}

function eosIds(model: Awaited<ReturnType<typeof Gemma4ForCausalLM.from_pretrained>>): Set<number> {
  const value = (model.config as unknown as { eos_token_id: number | readonly number[] }).eos_token_id;
  return new Set(Array.isArray(value) ? value : [value]);
}

function unavailableStage(name: LoadResult["stages"][number]["name"], note: string) {
  return { name, durationMs: null, observable: false, note } as const;
}

function required<T>(value: T | null, message: string): T {
  if (value === null) throw new Error(message);
  return value;
}

function emptyCallbacks(): GenerationCallbacks {
  return { onRequestStart() {}, onTextChunk() {} };
}

function disposeTensorValues(value: unknown): void {
  if (value === null || typeof value !== "object") return;
  if ("dispose" in value && typeof value.dispose === "function") {
    value.dispose();
    return;
  }
  for (const child of Object.values(value)) if (child !== value) disposeTensorValues(child);
}