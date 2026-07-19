import type {
  BenchmarkAdapter,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
} from "../types";

const SOURCE_REVISION = "158f16ae0f672943ca304d59c47c8e3a264e399e";
const MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers";
const MODEL_REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";
const BUNDLE_URL = `https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels/resolve/${SOURCE_REVISION}/gemma-4-e2b.js`;

type ChatMessage = { role: "user" | "assistant"; content: string };

interface PinnedGemmaModel {
  encodePrompt(messages: readonly ChatMessage[]): number[];
  generate(
    messages: readonly ChatMessage[],
    options: { maxNewTokens: number },
  ): AsyncIterable<{ token: number; delta: string; text: string }>;
  deviceInfo(): { features: { shaderF16: boolean; subgroups: boolean } };
  reset(): void;
  dispose(): void;
}

interface PinnedGemmaModule {
  Gemma4Mobile: {
    load(
      modelId?: string | null,
      options?: { revision?: string; fetch?: typeof fetch },
    ): Promise<PinnedGemmaModel>;
  };
}

export class PinnedHuggingFaceBenchmarkAdapter implements BenchmarkAdapter {
  readonly id = "pinned-hugging-face-webgpu";
  readonly runtimeName = "Pinned Hugging Face WebGPU";
  readonly runtimeVersion = SOURCE_REVISION;
  readonly modelId = MODEL_ID;
  readonly modelRevision = MODEL_REVISION;
  readonly artifactType = "mobile-QAT safetensors";
  readonly artifactUrl = `https://huggingface.co/${MODEL_ID}/resolve/${MODEL_REVISION}/model.safetensors`;
  readonly artifactBytes = 2_458_111_846;
  readonly artifactEquivalence = "pinned-source-equivalent" as const;
  readonly available = true;
  readonly limitations = Object.freeze([
    `The runtime is loaded from the immutable upstream bundle at ${SOURCE_REVISION}.`,
    "The upstream API exposes aggregate readiness only, not separate parse, graph creation, GPU upload, or shader compilation stages.",
    "The upstream runtime fixes its default KV cache capacity at 8,192 tokens.",
  ]);

  private model: PinnedGemmaModel | null = null;
  private messages: ChatMessage[] = [];
  private conversationEnabled = false;
  private lastGeneratedText = "";
  private lastGeneratedTokenCount = 0;

  async load(options: LoadOptions): Promise<LoadResult> {
    await this.dispose();
    const startedAtMs = performance.now();
    const module = await importPinnedModule();
    this.model = await module.Gemma4Mobile.load(null, {
      revision: MODEL_REVISION,
      ...(options.sourceUrl ? { fetch: localSnapshotFetch(options.sourceUrl) } : {}),
    });
    const readyAtMs = performance.now();
    const device = this.model.deviceInfo();
    return {
      startedAtMs,
      readyAtMs,
      readyMs: readyAtMs - startedAtMs,
      bytesTransferred: 0,
      stages: [
        unavailableStage("model-download", "Measured by the browser network observer."),
        unavailableStage("cached-asset-read", "Included in aggregate ready time."),
        unavailableStage("parse-deserialize", "Not separately exposed by the upstream bundle."),
        unavailableStage("graph-create", "Not separately exposed by the upstream bundle."),
        unavailableStage("gpu-upload", "Not separately exposed by the upstream bundle."),
        unavailableStage("shader-compile", "Not separately exposed by the upstream bundle."),
        { name: "ready", durationMs: readyAtMs - startedAtMs, observable: true },
      ],
      webgpuVerified: device.features.shaderF16 && device.features.subgroups,
      backend: "pinned-hugging-face-webgpu",
      memoryBytes: null,
      notes: this.limitations,
    };
  }

  async warmup(testCase: BenchmarkCase): Promise<void> {
    await this.generate(testCase, { onRequestStart() {}, onTextChunk() {} });
  }

  async generate(testCase: BenchmarkCase, callbacks: GenerationCallbacks): Promise<GenerationResult> {
    const model = required(this.model, "Pinned Hugging Face benchmark adapter is not loaded");
    const messages = this.conversationEnabled
      ? [...this.messages, userMessage(testCase.prompt)]
      : [userMessage(testCase.prompt)];
    const inputTokens = model.encodePrompt(messages).length;
    const outputTokenIds: number[] = [];
    let text = "";
    callbacks.onRequestStart(performance.now());
    for await (const update of model.generate(messages, { maxNewTokens: testCase.targetOutputTokens })) {
      outputTokenIds.push(update.token);
      text = update.text;
      callbacks.onTextChunk(update.delta, performance.now());
    }
    this.lastGeneratedText = text;
    this.lastGeneratedTokenCount = outputTokenIds.length;
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
    const model = required(this.model, "Pinned Hugging Face benchmark adapter is not loaded");
    if (purpose === "output" && text === this.lastGeneratedText) return this.lastGeneratedTokenCount;
    return model.encodePrompt([userMessage(text)]).length;
  }

  async resetConversation(): Promise<void> {
    this.messages = [];
    this.conversationEnabled = false;
    this.model?.reset();
  }

  async createConversation(): Promise<void> {
    this.messages = [];
    this.conversationEnabled = true;
    this.model?.reset();
  }

  async dispose(): Promise<void> {
    this.model?.dispose();
    this.model = null;
    this.messages = [];
    this.conversationEnabled = false;
    this.lastGeneratedText = "";
    this.lastGeneratedTokenCount = 0;
  }
}

function userMessage(content: string): ChatMessage {
  return { role: "user", content };
}

function assistantMessage(content: string): ChatMessage {
  return { role: "assistant", content };
}

function unavailableStage(name: LoadResult["stages"][number]["name"], note: string) {
  return { name, durationMs: null, observable: false, note } as const;
}

function required<T>(value: T | null, message: string): T {
  if (value === null) throw new Error(message);
  return value;
}

async function importPinnedModule(): Promise<PinnedGemmaModule> {
  const response = await fetch(BUNDLE_URL);
  if (!response.ok) throw new Error(`Pinned Hugging Face bundle request failed: ${response.status}`);
  const source = await response.text();
  const moduleUrl = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  try {
    return await import(/* @vite-ignore */ moduleUrl) as PinnedGemmaModule;
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }
}

function localSnapshotFetch(sourceUrl: string): typeof fetch {
  const modelUrl = new URL(sourceUrl, location.href).toString();
  const localResources = new Map([
    ["model.safetensors", modelUrl],
    ["config.json", new URL("/models/gemma-4-e2b-tokenizer/config.json", location.href).toString()],
    ["tokenizer.json", new URL("/models/gemma-4-e2b-tokenizer/tokenizer.json", location.href).toString()],
    ["tokenizer_config.json", new URL("/models/gemma-4-e2b-tokenizer/tokenizer_config.json", location.href).toString()],
    ["generation_config.json", new URL("/models/gemma-4-e2b-tokenizer/generation_config.json", location.href).toString()],
  ]);
  return async (input, init) => {
    const requested = new URL(input instanceof Request ? input.url : String(input), location.href);
    const fileName = requested.pathname.split("/").at(-1) ?? "";
    const mapped = localResources.get(fileName);
    const response = await globalThis.fetch(mapped ?? input, init);
    const method = init?.method ?? (input instanceof Request ? input.method : "GET");
    if (mapped && fileName === "model.safetensors" && method.toUpperCase() === "HEAD") {
      const headers = new Headers(response.headers);
      headers.set("Accept-Ranges", "bytes");
      return new Response(null, { status: response.status, statusText: response.statusText, headers });
    }
    return response;
  };
}