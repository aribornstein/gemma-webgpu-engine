import { GEMMA_GREEDY_GOLDEN_CASES } from "../runtime/gemma-golden";
import {
  loadGemmaGenerationSession,
  type GemmaGenerationResult,
  type GemmaGenerationSession,
  type GemmaGenerationTiming,
  type GemmaSessionMemoryEstimate,
} from "../runtime/gemma-session";

export type ReliabilitySmokeScenario =
  | "golden"
  | "sampling"
  | "regex-constraint"
  | "json-constraint"
  | "prefix-reuse"
  | "cancellation-recovery"
  | "lifecycle-reload"
  | "device-loss-recovery"
  | "vision"
  | "audio";

export interface ReliabilitySmokeInitialization {
  loadMs: number;
  localModelAvailable: boolean;
  memory: GemmaSessionMemoryEstimate;
}

export interface ReliabilitySmokeScenarioResult {
  scenario: ReliabilitySmokeScenario;
  durationMs: number;
  details: Readonly<Record<string, unknown>>;
  memory: GemmaSessionMemoryEstimate;
}

let session: GemmaGenerationSession | null = null;
let sourceUrl: string | undefined;
let cacheCapacity = 1024;

export async function initializeReliabilitySmoke(
  options: { sourceUrl?: string; cacheCapacity?: number } = {},
): Promise<ReliabilitySmokeInitialization> {
  await disposeReliabilitySmoke();
  sourceUrl = options.sourceUrl;
  cacheCapacity = options.cacheCapacity ?? 1024;
  const localModelAvailable = sourceUrl
    ? (await fetch(sourceUrl, { method: "HEAD" })).ok
    : false;
  if (sourceUrl && !localModelAvailable) {
    throw new Error(`Reliability smoke model source is unavailable: ${sourceUrl}`);
  }
  const startedAt = performance.now();
  session = await loadSession();
  return {
    loadMs: performance.now() - startedAt,
    localModelAvailable,
    memory: session.estimateRetainedGpuMemory(),
  };
}

export async function executeReliabilitySmokeScenario(
  scenario: ReliabilitySmokeScenario,
  sequence: number,
): Promise<ReliabilitySmokeScenarioResult> {
  const startedAt = performance.now();
  const details = await executeScenario(scenario, sequence);
  return {
    scenario,
    durationMs: performance.now() - startedAt,
    details,
    memory: requiredSession().estimateRetainedGpuMemory(),
  };
}

export async function disposeReliabilitySmoke(): Promise<void> {
  session?.destroy();
  session = null;
}

async function executeScenario(
  scenario: ReliabilitySmokeScenario,
  sequence: number,
): Promise<Readonly<Record<string, unknown>>> {
  switch (scenario) {
    case "golden":
      return executeGolden(sequence);
    case "sampling":
      return executeSampling();
    case "regex-constraint":
      return executeRegexConstraint();
    case "json-constraint":
      return executeJsonConstraint();
    case "prefix-reuse":
      return executePrefixReuse();
    case "cancellation-recovery":
      return executeCancellationRecovery();
    case "lifecycle-reload":
      return executeLifecycleReload();
    case "device-loss-recovery":
      return executeDeviceLossRecovery();
    case "vision":
      return executeVision();
    case "audio":
      return executeAudio();
  }
}

async function executeGolden(sequence: number): Promise<Readonly<Record<string, unknown>>> {
  const golden = GEMMA_GREEDY_GOLDEN_CASES[sequence % GEMMA_GREEDY_GOLDEN_CASES.length];
  const measured = await requiredSession().generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
    reusePromptCache: false,
  });
  assertGolden(measured.result, golden);
  return {
    goldenId: golden.id,
    text: measured.result.text,
    generatedTokenIds: measured.result.generatedTokenIds,
    timing: summarizeTiming(measured.timing),
  };
}

async function executeSampling(): Promise<Readonly<Record<string, unknown>>> {
  const prompt = "Write one vivid sentence describing a pocket-sized observatory.";
  const options = {
    maxNewTokens: 32,
    temperature: 0.8,
    topK: 40,
    topP: 0.9,
    minP: 0.05,
    typicalP: 0.95,
    repetitionPenalty: 1.08,
    seed: 7,
    reusePromptCache: false,
  } as const;
  const first = await requiredSession().generateMeasured(prompt, options);
  const second = await requiredSession().generateMeasured(prompt, options);
  assertEqual(first.result.text, second.result.text, "Seeded sampling text changed");
  assertNumberArraysEqual(
    first.result.generatedTokenIds,
    second.result.generatedTokenIds,
    "Seeded sampling token IDs changed",
  );
  return {
    text: first.result.text,
    generatedTokenIds: first.result.generatedTokenIds,
    firstTiming: summarizeTiming(first.timing),
    secondTiming: summarizeTiming(second.timing),
  };
}

async function executeRegexConstraint(): Promise<Readonly<Record<string, unknown>>> {
  const measured = await requiredSession().generateMeasured(
    "Return exactly one lowercase word for the color of a clear daytime sky.",
    {
      maxNewTokens: 4,
      constraint: { type: "regex", pattern: "(?:blue|gray)" },
      reusePromptCache: false,
    },
  );
  if (!/^(blue|gray)$/.test(measured.result.text)) {
    throw new Error(`Regex-constrained output is invalid: ${measured.result.text}`);
  }
  return { text: measured.result.text, timing: summarizeTiming(measured.timing) };
}

async function executeJsonConstraint(): Promise<Readonly<Record<string, unknown>>> {
  const measured = await requiredSession().generateMeasured(
    "Return one compact JSON object with exactly the key ok and the boolean value true.",
    {
      maxNewTokens: 16,
      constraint: { type: "json", maxDepth: 2, whitespace: "compact" },
      reusePromptCache: false,
    },
  );
  const parsed = JSON.parse(measured.result.text) as Record<string, unknown>;
  assertEqual(parsed.ok, true, "JSON-constrained output did not contain ok=true");
  assertEqual(Object.keys(parsed).length, 1, "JSON-constrained output contained extra keys");
  return { text: measured.result.text, timing: summarizeTiming(measured.timing) };
}

async function executePrefixReuse(): Promise<Readonly<Record<string, unknown>>> {
  const golden = GEMMA_GREEDY_GOLDEN_CASES[3];
  const first = await requiredSession().generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
    reusePromptCache: false,
  });
  const second = await requiredSession().generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
    reusePromptCache: true,
  });
  assertGolden(first.result, golden);
  assertGolden(second.result, golden);
  const expectedReuse = golden.promptTokenIds.length - 1;
  assertEqual(
    second.timing.promptTokensReused,
    expectedReuse,
    "Prompt cache did not retain the exact reusable prefix",
  );
  return {
    promptTokensReused: second.timing.promptTokensReused,
    freshTiming: summarizeTiming(first.timing),
    reusedTiming: summarizeTiming(second.timing),
  };
}

async function executeCancellationRecovery(): Promise<Readonly<Record<string, unknown>>> {
  const controller = new AbortController();
  let emittedTokens = 0;
  let cancellation: { name: string; message: string } | null = null;
  try {
    await requiredSession().generate(
      "Write a detailed paragraph naming and comparing the primary colors.",
      {
        maxNewTokens: 64,
        reusePromptCache: false,
        signal: controller.signal,
        onToken() {
          emittedTokens += 1;
          if (emittedTokens === 1) {
            controller.abort(new DOMException("reliability smoke cancellation", "AbortError"));
          }
        },
      },
    );
  } catch (error) {
    cancellation = {
      name: error instanceof Error ? error.name : "Error",
      message: error instanceof Error ? error.message : String(error),
    };
  }
  assertEqual(cancellation?.name, "AbortError", "Generation did not cancel with AbortError");
  assertEqual(emittedTokens, 1, "Cancellation emitted an unexpected number of tokens");
  const golden = GEMMA_GREEDY_GOLDEN_CASES[0];
  const recovery = await requiredSession().generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
  });
  assertGolden(recovery.result, golden);
  assertEqual(recovery.timing.promptTokensReused, 0, "Cancelled cache state was reused");
  return {
    emittedTokens,
    cancellation,
    recoveryText: recovery.result.text,
    recoveryTiming: summarizeTiming(recovery.timing),
  };
}

async function executeLifecycleReload(): Promise<Readonly<Record<string, unknown>>> {
  const previous = requiredSession();
  previous.destroy();
  previous.destroy();
  let destroyedSessionError: string | null = null;
  try {
    await previous.generate("Say hi.", { maxNewTokens: 1 });
  } catch (error) {
    destroyedSessionError = error instanceof Error ? error.message : String(error);
  }
  if (!destroyedSessionError?.includes("destroyed")) {
    throw new Error("Destroyed session accepted generation");
  }
  const startedAt = performance.now();
  session = await loadSession();
  const loadMs = performance.now() - startedAt;
  const golden = GEMMA_GREEDY_GOLDEN_CASES[0];
  const recovery = await session.generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
  });
  assertGolden(recovery.result, golden);
  return { destroyedSessionError, loadMs, recoveryText: recovery.result.text };
}

async function executeDeviceLossRecovery(): Promise<Readonly<Record<string, unknown>>> {
  const previous = requiredSession();
  previous.simulateDeviceLoss();
  const loss = await previous.deviceLost;
  previous.destroy();
  const startedAt = performance.now();
  session = await loadSession();
  const loadMs = performance.now() - startedAt;
  const golden = GEMMA_GREEDY_GOLDEN_CASES[1];
  const recovery = await session.generateMeasured(golden.prompt, {
    maxNewTokens: golden.maxNewTokens,
  });
  assertGolden(recovery.result, golden);
  return {
    lossReason: loss.reason,
    lossMessage: loss.message,
    loadMs,
    recoveryText: recovery.result.text,
  };
}

async function executeVision(): Promise<Readonly<Record<string, unknown>>> {
  const image = await fetchBlob("/examples/dolphin_capt_image.png");
  const measured = await requiredSession().generateMeasured({
    messages: [{
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Describe this image briefly." }],
    }],
    images: [image],
    visionTokenBudget: 70,
  }, { maxNewTokens: 24, reusePromptCache: false });
  if (!measured.result.text.trim()) throw new Error("Vision request returned empty text");
  if (measured.timing.visionEncodeMs <= 0) throw new Error("Vision request skipped encoding");
  return { text: measured.result.text, timing: summarizeTiming(measured.timing) };
}

async function executeAudio(): Promise<Readonly<Record<string, unknown>>> {
  const audio = await fetchBlob("/examples/gemma-audio-demo.wav");
  const measured = await requiredSession().generateMeasured({
    messages: [{
      role: "user",
      content: [{ type: "audio" }, {
        type: "text",
        text: "What is said in this audio? Return only the spoken words.",
      }],
    }],
    audios: [audio],
  }, { maxNewTokens: 32, reusePromptCache: false });
  if (!/web gpu audio is working in your browser/i.test(measured.result.text)) {
    throw new Error(`Audio transcription changed: ${measured.result.text}`);
  }
  if (measured.timing.audioEncodeMs <= 0) throw new Error("Audio request skipped encoding");
  return { text: measured.result.text, timing: summarizeTiming(measured.timing) };
}

function loadSession(): Promise<GemmaGenerationSession> {
  return loadGemmaGenerationSession({ cacheCapacity, sourceUrl, prefillStrategy: "auto" });
}

function requiredSession(): GemmaGenerationSession {
  if (!session) throw new Error("Reliability smoke session is not initialized");
  return session;
}

async function fetchBlob(url: string): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Reliability fixture is unavailable: ${url}`);
  return response.blob();
}

function assertGolden(
  result: GemmaGenerationResult,
  golden: (typeof GEMMA_GREEDY_GOLDEN_CASES)[number],
): void {
  assertEqual(result.text, golden.text, `Golden ${golden.id} text changed`);
  assertNumberArraysEqual(
    result.generatedTokenIds,
    golden.generatedTokenIds,
    `Golden ${golden.id} token IDs changed`,
  );
  assertEqual(
    result.stoppedOnEndToken,
    golden.stoppedOnEndToken,
    `Golden ${golden.id} stop behavior changed`,
  );
}

function summarizeTiming(timing: GemmaGenerationTiming): Readonly<Record<string, unknown>> {
  return {
    totalMs: timing.totalMs,
    timeToFirstTokenMs: timing.timeToFirstTokenMs,
    prefillMs: timing.prefillMs,
    promptTokensReused: timing.promptTokensReused,
    decodeTokenCount: timing.decodeTokenMs.length,
    logitsReadbackMs: timing.logitsReadbackMs,
    visionEncodeMs: timing.visionEncodeMs,
    audioEncodeMs: timing.audioEncodeMs,
  };
}

function assertNumberArraysEqual(
  actual: readonly number[],
  expected: readonly number[],
  message: string,
): void {
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw new Error(`${message}: ${JSON.stringify(actual)} !== ${JSON.stringify(expected)}`);
  }
}

function assertEqual(actual: unknown, expected: unknown, message: string): void {
  if (!Object.is(actual, expected)) {
    throw new Error(`${message}: ${String(actual)} !== ${String(expected)}`);
  }
}