import {
  AutoProcessor,
  Gemma4ForCausalLM,
  TextStreamer,
  env,
} from "@huggingface/transformers";

const TRANSFORMERS_JS_E2B_MODEL_ID = "onnx-community/gemma-4-E2B-it-ONNX";
const TRANSFORMERS_JS_E2B_REVISION = "9f4bef82ea6e296bc69f8a2f5939f73af81b07a6";

export interface TransformersJsE2BBenchmarkCase {
  id: string;
  prompt: string;
  maxOutputTokens: number;
  expectedText?: string;
}

export interface TransformersJsE2BBenchmarkOptions {
  cases: readonly TransformersJsE2BBenchmarkCase[];
  warmupIterations?: number;
  iterations?: number;
}

export interface TransformersJsE2BSample {
  iteration: number;
  outputText: string;
  generatedTokenIds: readonly number[];
  exactTextMatch: boolean | null;
  timing: {
    timeToFirstTokenMs: number | null;
    interTokenLatencyMs: readonly number[];
    timePerOutputTokenMs: number | null;
    decodeTokensPerSecond: number | null;
    totalMs: number;
  };
}

export interface TransformersJsE2BCaseArtifact {
  id: string;
  prompt: string;
  promptTokens: number;
  maxOutputTokens: number;
  samples: readonly TransformersJsE2BSample[];
}

export interface TransformersJsE2BBenchmarkArtifact {
  schemaVersion: 1;
  capturedAt: string;
  runtime: "@huggingface/transformers";
  runtimeVersion: "4.2.0";
  modelId: string;
  modelRevision: string;
  modelVariant: "q4f16";
  modelEquivalence: "model-family-only";
  loadMs: number;
  environment: {
    userAgent: string;
    adapterInfo: Record<string, string>;
  };
  configuration: {
    warmupIterations: number;
    iterations: number;
    sampler: "greedy";
    textOnly: true;
    freshGenerationStatePerSample: true;
  };
  cases: readonly TransformersJsE2BCaseArtifact[];
  limitations: readonly string[];
}

export async function benchmarkTransformersJsE2B(
  options: TransformersJsE2BBenchmarkOptions,
): Promise<TransformersJsE2BBenchmarkArtifact> {
  if (options.cases.length === 0) {
    throw new Error("Transformers.js benchmark requires cases");
  }
  const warmupIterations = nonNegativeInteger(
    options.warmupIterations ?? 1,
    "warmup iterations",
  );
  const iterations = positiveInteger(options.iterations ?? 3, "iterations");
  const previousEnvironment = {
    allowLocalModels: env.allowLocalModels,
    allowRemoteModels: env.allowRemoteModels,
    useBrowserCache: env.useBrowserCache,
  };
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  let model: Awaited<ReturnType<typeof Gemma4ForCausalLM.from_pretrained>> | undefined;

  try {
  const adapterInfoPromise = readAdapterInfo();
  const loadStartedAt = performance.now();
  const processor = await AutoProcessor.from_pretrained(
    TRANSFORMERS_JS_E2B_MODEL_ID,
    { revision: TRANSFORMERS_JS_E2B_REVISION },
  );
  model = await Gemma4ForCausalLM.from_pretrained(
    TRANSFORMERS_JS_E2B_MODEL_ID,
    {
      revision: TRANSFORMERS_JS_E2B_REVISION,
      device: "webgpu",
      dtype: "q4f16",
    },
  );
  const loadMs = performance.now() - loadStartedAt;

    const cases: TransformersJsE2BCaseArtifact[] = [];
    for (const testCase of options.cases) {
      const prompt = formatPrompt(processor, testCase.prompt);
      const promptInputs = await processor(prompt, null, null, {
        add_special_tokens: false,
      });
      const promptTokens = promptInputs.input_ids.dims.at(-1);
      disposeTensorValues(promptInputs);
      if (promptTokens === undefined) {
        throw new Error("Transformers.js tokenizer did not return a prompt length");
      }

      for (let iteration = 0; iteration < warmupIterations; iteration += 1) {
        await runSample(processor, model, testCase, iteration);
      }
      const samples: TransformersJsE2BSample[] = [];
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        samples.push(await runSample(processor, model, testCase, iteration));
      }
      cases.push({
        id: testCase.id,
        prompt: testCase.prompt,
        promptTokens,
        maxOutputTokens: testCase.maxOutputTokens,
        samples: Object.freeze(samples),
      });
    }

    return {
      schemaVersion: 1,
      capturedAt: new Date().toISOString(),
      runtime: "@huggingface/transformers",
      runtimeVersion: "4.2.0",
      modelId: TRANSFORMERS_JS_E2B_MODEL_ID,
      modelRevision: TRANSFORMERS_JS_E2B_REVISION,
      modelVariant: "q4f16",
      modelEquivalence: "model-family-only",
      loadMs: round(loadMs),
      environment: {
        userAgent: navigator.userAgent,
        adapterInfo: await adapterInfoPromise,
      },
      configuration: {
        warmupIterations,
        iterations,
        sampler: "greedy",
        textOnly: true,
        freshGenerationStatePerSample: true,
      },
      cases: Object.freeze(cases),
      limitations: Object.freeze([
        "The ONNX Community q4f16 export is derived from the Gemma 4 E2B instruction model, but it is not the file-identical mixed 2/4/8-bit mobile-QAT safetensors artifact used by the owned runtime.",
        "Gemma4ForCausalLM intentionally selects only embed_tokens and decoder_model_merged; vision and audio encoder sessions are excluded from load time and generation.",
        "Token timestamps are captured by TextStreamer after each generated token reaches JavaScript. They include runtime dispatch, readback, sampling, and callback overhead.",
      ]),
    };
  } finally {
    if (model) await model.dispose();
    env.allowLocalModels = previousEnvironment.allowLocalModels;
    env.allowRemoteModels = previousEnvironment.allowRemoteModels;
    env.useBrowserCache = previousEnvironment.useBrowserCache;
  }
}

async function runSample(
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>,
  model: Awaited<ReturnType<typeof Gemma4ForCausalLM.from_pretrained>>,
  testCase: TransformersJsE2BBenchmarkCase,
  iteration: number,
): Promise<TransformersJsE2BSample> {
  const startedAt = performance.now();
  const prompt = formatPrompt(processor, testCase.prompt);
  const inputs = await processor(prompt, null, null, { add_special_tokens: false });
  const generatedTokenIds: number[] = [];
  const tokenTimestamps: number[] = [];
  const generationTimestamps: number[] = [];
  const endTokenIds = (model.config as unknown as {
    eos_token_id: number | readonly number[];
  }).eos_token_id;
  const endTokens = new Set<number>(Array.isArray(endTokenIds) ? endTokenIds : [endTokenIds]);
  const streamer = new TextStreamer(processor.tokenizer!, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: () => {},
    token_callback_function: (tokens) => {
      const observedAt = performance.now();
      for (const token of tokens) {
        const tokenId = Number(token);
        generationTimestamps.push(observedAt);
        if (endTokens.has(tokenId)) continue;
        generatedTokenIds.push(tokenId);
        tokenTimestamps.push(observedAt);
      }
    },
  });
  try {
    const output = await model.generate({
      ...inputs,
      max_new_tokens: positiveInteger(testCase.maxOutputTokens, "max output tokens"),
      do_sample: false,
      streamer,
    });
    disposeTensorValues(output);
  } finally {
    disposeTensorValues(inputs);
  }

  const totalMs = performance.now() - startedAt;
  const interTokenLatencyMs = tokenTimestamps.slice(1).map((timestamp, index) =>
    round(timestamp - tokenTimestamps[index])
  );
  const decodeStepMs = generationTimestamps.slice(1).map((timestamp, index) =>
    timestamp - generationTimestamps[index]
  );
  const totalDecodeMs = decodeStepMs.reduce((sum, value) => sum + value, 0);
  const totalInterTokenMs = interTokenLatencyMs.reduce((sum, value) => sum + value, 0);
  const timePerOutputTokenMs = interTokenLatencyMs.length === 0
    ? null
    : totalInterTokenMs / interTokenLatencyMs.length;
  const outputText = processor.decode(generatedTokenIds, { skip_special_tokens: true });
  return {
    iteration,
    outputText,
    generatedTokenIds: Object.freeze(generatedTokenIds),
    exactTextMatch: testCase.expectedText === undefined
      ? null
      : outputText === testCase.expectedText,
    timing: {
      timeToFirstTokenMs: generationTimestamps.length === 0
        ? null
        : round(generationTimestamps[0] - startedAt),
      interTokenLatencyMs: Object.freeze(interTokenLatencyMs),
      timePerOutputTokenMs: timePerOutputTokenMs === null
        ? null
        : round(timePerOutputTokenMs),
      decodeTokensPerSecond: totalDecodeMs > 0
        ? round(decodeStepMs.length * 1000 / totalDecodeMs)
        : null,
      totalMs: round(totalMs),
    },
  };
}

function formatPrompt(
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>,
  prompt: string,
): string {
  const formatted = processor.apply_chat_template(
    [{ role: "user", content: [{ type: "text", text: prompt }] }],
    { add_generation_prompt: true, tokenize: false },
  );
  if (typeof formatted !== "string") {
    throw new Error("Transformers.js chat template did not return text");
  }
  return formatted;
}

function disposeTensorValues(value: unknown): void {
  if (value === null || typeof value !== "object") return;
  if ("dispose" in value && typeof value.dispose === "function") {
    value.dispose();
    return;
  }
  for (const child of Object.values(value)) {
    if (child !== value) disposeTensorValues(child);
  }
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be an integer >= 1`);
  }
  return value;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label} must be an integer >= 0`);
  }
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