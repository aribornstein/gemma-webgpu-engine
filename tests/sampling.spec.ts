import { expect, test } from "@playwright/test";
import { DEFAULT_DECODING_CONFIG, validateDecodingConfig } from "../src/runtime/decoding";
import {
  resolveGemmaGenerationConfig,
  usesGemmaGpuGreedy,
} from "../src/runtime/generation-config";
import {
  emitGemmaGenerationUpdate,
  throwIfGemmaGenerationAborted,
} from "../src/runtime/generation-control";
import { sampleToken, SeededRandom } from "../src/runtime/sampling";

test("temperature zero performs greedy decoding", () => {
  expect(sampleToken([0.2, 4, 1], [], { ...DEFAULT_DECODING_CONFIG, temperature: 0 }, () => 0.5)).toBe(1);
});

test("top-k one is deterministic", () => {
  expect(sampleToken([0.2, 4, 1], [], { ...DEFAULT_DECODING_CONFIG, topK: 1 }, () => 0.99)).toBe(1);
});

test("seeded sampling is reproducible", () => {
  const first = new SeededRandom(71);
  const second = new SeededRandom(71);
  const config = { ...DEFAULT_DECODING_CONFIG, topK: 0, topP: 1 };
  const a = Array.from({ length: 20 }, () => sampleToken([1, 1, 1], [], config, () => first.next()));
  const b = Array.from({ length: 20 }, () => sampleToken([1, 1, 1], [], config, () => second.next()));
  expect(a).toEqual(b);
});

test("repetition penalty can move selection away from history", () => {
  expect(sampleToken([2, 1.5], [0], { ...DEFAULT_DECODING_CONFIG, temperature: 0, repetitionPenalty: 2 }, () => 0)).toBe(1);
});

test("zero repetition window disables history penalties", () => {
  expect(sampleToken(
    [2, 1.5],
    [0],
    {
      ...DEFAULT_DECODING_CONFIG,
      temperature: 0,
      repetitionPenalty: 2,
      repetitionWindow: 0,
    },
    () => 0,
  )).toBe(0);
});

test("validates every decoding control", () => {
  const result = validateDecodingConfig({
    topP: Number.NaN,
    repetitionWindow: -1,
    frequencyPenalty: Number.POSITIVE_INFINITY,
    presencePenalty: -0.1,
    stopTokenIds: [1, -2],
  });
  expect(result.ok).toBe(false);
  expect(result.errors).toEqual([
    "topP must be between 0 and 1",
    "repetitionWindow must be a non-negative integer",
    "frequencyPenalty must be finite and non-negative",
    "presencePenalty must be finite and non-negative",
    "stopTokenIds must contain non-negative integers",
  ]);
});

test("generation defaults preserve exact GPU greedy decoding", () => {
  const config = resolveGemmaGenerationConfig({});
  expect(config.maxNewTokens).toBe(32);
  expect(config.temperature).toBe(0);
  expect(config.repetitionPenalty).toBe(1);
  expect(usesGemmaGpuGreedy(config)).toBe(true);
});

test("generation routes sampling and penalties through logits readback", () => {
  expect(usesGemmaGpuGreedy(resolveGemmaGenerationConfig({ temperature: 0.8 }))).toBe(false);
  expect(usesGemmaGpuGreedy(resolveGemmaGenerationConfig({ repetitionPenalty: 1.1 }))).toBe(false);
  expect(usesGemmaGpuGreedy(resolveGemmaGenerationConfig({ presencePenalty: 0.1 }))).toBe(false);
});

test("generation rejects stop token IDs outside the Gemma vocabulary", () => {
  expect(() => resolveGemmaGenerationConfig({ stopTokenIds: [262144] })).toThrow(
    "Gemma stop token IDs must be below 262144",
  );
});

test("presence and frequency penalties use counts inside the repetition window", () => {
  const config = {
    ...DEFAULT_DECODING_CONFIG,
    temperature: 0,
    repetitionPenalty: 1,
    repetitionWindow: 2,
    presencePenalty: 0.1,
    frequencyPenalty: 0.3,
  };
  expect(sampleToken([2, 1.5], [0, 0], config, () => 0)).toBe(1);

  expect(sampleToken(
    [2, 2.1],
    [0, 1],
    { ...config, repetitionWindow: 1, presencePenalty: 0.5, frequencyPenalty: 0 },
    () => 0,
  )).toBe(0);
});

test("min-p filters tokens relative to the leading probability", () => {
  const config = {
    ...DEFAULT_DECODING_CONFIG,
    temperature: 1,
    topK: 0,
    topP: 1,
    typicalP: 1,
    minP: 0.6,
  };
  expect(sampleToken([Math.log(0.6), Math.log(0.3), Math.log(0.1)], [], config, () => 0.99)).toBe(0);
});

test("top-p keeps the smallest leading probability mass", () => {
  const config = {
    ...DEFAULT_DECODING_CONFIG,
    temperature: 1,
    topK: 0,
    topP: 0.5,
    minP: 0,
    typicalP: 1,
  };
  expect(sampleToken([Math.log(0.6), Math.log(0.3), Math.log(0.1)], [], config, () => 0.99)).toBe(0);
});

test("typical-p can select a non-leading token closest to entropy", () => {
  const config = {
    ...DEFAULT_DECODING_CONFIG,
    temperature: 1,
    topK: 0,
    topP: 1,
    minP: 0,
    typicalP: 0.3,
  };
  expect(sampleToken([Math.log(0.45), Math.log(0.4), Math.log(0.15)], [], config, () => 0.5)).toBe(1);
});

test("streams immutable accumulated token updates with backpressure", async () => {
  const generatedTokenIds = [10, 20];
  let handlerCompleted = false;
  await emitGemmaGenerationUpdate(
    20,
    generatedTokenIds,
    (tokenIds) => tokenIds.join(" "),
    async (update) => {
      expect(update).toEqual({
        tokenId: 20,
        tokenIndex: 1,
        generatedTokenIds: [10, 20],
        text: "10 20",
      });
      generatedTokenIds.push(30);
      await Promise.resolve();
      expect(update.generatedTokenIds).toEqual([10, 20]);
      handlerCompleted = true;
    },
  );
  expect(handlerCompleted).toBe(true);
});

test("throws the AbortSignal reason when generation is cancelled", () => {
  const controller = new AbortController();
  const reason = new Error("cancelled by test");
  controller.abort(reason);
  expect(() => throwIfGemmaGenerationAborted(controller.signal)).toThrow(reason);
  expect(() => throwIfGemmaGenerationAborted()).not.toThrow();
});
