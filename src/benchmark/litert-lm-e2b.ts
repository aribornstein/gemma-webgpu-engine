import {
  Engine,
  SamplerType,
  type Message,
} from "@litert-lm/core";

const LITERT_E2B_MODEL_URL =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm";

export interface LiteRtLmE2BBenchmarkCase {
  id: string;
  prompt: string;
  maxOutputTokens: number;
  expectedText?: string;
}

export interface LiteRtLmE2BBenchmarkOptions {
  cases: readonly LiteRtLmE2BBenchmarkCase[];
  contextCapacity?: number;
  warmupIterations?: number;
  iterations?: number;
  modelUrl?: string;
}

export interface LiteRtLmE2BSample {
  iteration: number;
  outputText: string;
  exactTextMatch: boolean | null;
  externallyObserved: {
    timeToFirstChunkMs: number | null;
    chunkIntervalMs: readonly number[];
    totalMs: number;
    chunkCount: number;
  };
  native: {
    prefillTokensPerSecond: number;
    prefillTokenCount: number;
    decodeTokensPerSecond: number;
    decodeTokenCount: number;
    timeToFirstTokenMs: number;
  };
}

export interface LiteRtLmE2BCaseArtifact {
  id: string;
  prompt: string;
  maxOutputTokens: number;
  samples: readonly LiteRtLmE2BSample[];
}

export interface LiteRtLmE2BBenchmarkArtifact {
  schemaVersion: 1;
  capturedAt: string;
  runtime: "@litert-lm/core";
  runtimeVersion: "0.14.0";
  modelUrl: string;
  modelEquivalence: "model-family-only";
  loadMs: number;
  environment: {
    userAgent: string;
    adapterInfo: Record<string, string>;
  };
  configuration: {
    contextCapacity: number;
    warmupIterations: number;
    iterations: number;
    sampler: "greedy";
    freshConversationPerSample: true;
  };
  cases: readonly LiteRtLmE2BCaseArtifact[];
  limitations: readonly string[];
}

export async function benchmarkLiteRtLmE2B(
  options: LiteRtLmE2BBenchmarkOptions,
): Promise<LiteRtLmE2BBenchmarkArtifact> {
  if (options.cases.length === 0) throw new Error("LiteRT-LM benchmark requires cases");
  const contextCapacity = positiveInteger(options.contextCapacity ?? 2048, "context capacity");
  const warmupIterations = nonNegativeInteger(options.warmupIterations ?? 1, "warmup iterations");
  const iterations = positiveInteger(options.iterations ?? 3, "iterations");
  const modelUrl = options.modelUrl ?? LITERT_E2B_MODEL_URL;
  const adapterInfoPromise = readAdapterInfo();
  const loadStartedAt = performance.now();
  const engine = await Engine.create({
    model: modelUrl,
    mainExecutorSettings: { maxNumTokens: contextCapacity },
    benchmarkEnabled: true,
  });
  const loadMs = performance.now() - loadStartedAt;

  try {
    const cases: LiteRtLmE2BCaseArtifact[] = [];
    for (const testCase of options.cases) {
      for (let iteration = 0; iteration < warmupIterations; iteration += 1) {
        await runSample(engine, testCase, iteration);
      }
      const samples: LiteRtLmE2BSample[] = [];
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        samples.push(await runSample(engine, testCase, iteration));
      }
      cases.push({
        id: testCase.id,
        prompt: testCase.prompt,
        maxOutputTokens: testCase.maxOutputTokens,
        samples: Object.freeze(samples),
      });
    }

    return {
      schemaVersion: 1,
      capturedAt: new Date().toISOString(),
      runtime: "@litert-lm/core",
      runtimeVersion: "0.14.0",
      modelUrl,
      modelEquivalence: "model-family-only",
      loadMs: round(loadMs),
      environment: {
        userAgent: navigator.userAgent,
        adapterInfo: await adapterInfoPromise,
      },
      configuration: {
        contextCapacity,
        warmupIterations,
        iterations,
        sampler: "greedy",
        freshConversationPerSample: true,
      },
      cases: Object.freeze(cases),
      limitations: Object.freeze([
        "The official Web .litertlm file is a specially optimized text-only export, not the pinned mobile-QAT safetensors file.",
        "Externally observed callback intervals are chunk intervals. They are not promoted to per-token ITL because the preview API does not guarantee one token per callback.",
        "Native LiteRT-LM TTFT and decode throughput are retained as supplemental runtime counters; cross-runtime claims use externally equivalent boundaries where available.",
      ]),
    };
  } finally {
    await engine.delete();
  }
}

async function runSample(
  engine: Engine,
  testCase: LiteRtLmE2BBenchmarkCase,
  iteration: number,
): Promise<LiteRtLmE2BSample> {
  const conversation = await engine.createConversation({
    sessionConfig: {
      maxOutputTokens: positiveInteger(testCase.maxOutputTokens, "max output tokens"),
      samplerParams: { type: SamplerType.GREEDY },
    },
  });
  try {
    const startedAt = performance.now();
    let firstChunkAt: number | null = null;
    let previousChunkAt: number | null = null;
    let outputText = "";
    const chunkIntervalMs: number[] = [];
    let chunkCount = 0;
    const stream = conversation.sendMessageStreaming({ role: "user", content: testCase.prompt });
    for await (const message of stream) {
      const now = performance.now();
      const text = messageText(message);
      if (text.length === 0) continue;
      firstChunkAt ??= now;
      if (previousChunkAt !== null) chunkIntervalMs.push(round(now - previousChunkAt));
      previousChunkAt = now;
      outputText = mergeStreamText(outputText, text);
      chunkCount += 1;
    }
    const totalMs = performance.now() - startedAt;
    const native = await conversation.getBenchmarkInfo();
    return {
      iteration,
      outputText,
      exactTextMatch: testCase.expectedText === undefined
        ? null
        : outputText === testCase.expectedText,
      externallyObserved: {
        timeToFirstChunkMs: firstChunkAt === null ? null : round(firstChunkAt - startedAt),
        chunkIntervalMs: Object.freeze(chunkIntervalMs),
        totalMs: round(totalMs),
        chunkCount,
      },
      native: {
        prefillTokensPerSecond: round(native.lastPrefillTokensPerSecond),
        prefillTokenCount: native.lastPrefillTokenCount,
        decodeTokensPerSecond: round(native.lastDecodeTokensPerSecond),
        decodeTokenCount: native.lastDecodeTokenCount,
        timeToFirstTokenMs: round(native.timeToFirstTokenInSecond * 1000),
      },
    };
  } finally {
    await conversation.delete();
  }
}

function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  return (message.content ?? []).flatMap((part) =>
    part.type === "text" ? [part.text] : []
  ).join("");
}

function mergeStreamText(accumulated: string, next: string): string {
  if (next.startsWith(accumulated)) return next;
  if (accumulated.endsWith(next)) return accumulated;
  return accumulated + next;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${label} must be an integer >= 1`);
  return value;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0) throw new Error(`${label} must be an integer >= 0`);
  return value;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

async function readAdapterInfo(): Promise<Record<string, string>> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return {};
  return Object.fromEntries(
    ["vendor", "architecture", "device", "description"].flatMap((key) => {
      const value = adapter.info[key as keyof GPUAdapterInfo];
      return typeof value === "string" && value.length > 0 ? [[key, value]] : [];
    }),
  );
}