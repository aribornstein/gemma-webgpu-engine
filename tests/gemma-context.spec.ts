import { expect, test } from "@playwright/test";
import {
  assertGemmaContextSupported,
  availableGemmaOutputTokens,
  GEMMA_MODEL_CONTEXT_CAPACITY,
  GEMMA_SLIDING_CACHE_CAPACITY,
  GEMMA_VALIDATED_CONTEXT_CAPACITY,
  planGemmaContextMemory,
} from "../src/runtime/gemma-context";

test("reports model, validated, and sliding context capacities", () => {
  expect(GEMMA_MODEL_CONTEXT_CAPACITY).toBe(131_072);
  expect(GEMMA_VALIDATED_CONTEXT_CAPACITY).toBe(32_768);
  expect(GEMMA_SLIDING_CACHE_CAPACITY).toBe(512);
});

test("plans heterogeneous cache memory through 128K", () => {
  expect(planGemmaContextMemory(8_192)).toMatchObject({
    slidingPhysicalCapacity: 512,
    fullPhysicalCapacity: 8_192,
    largestCacheBufferBytes: 16_777_216,
    kvCacheBytes: 113_246_208,
  });
  expect(planGemmaContextMemory(32_768).largestCacheBufferBytes).toBe(67_108_864);
  expect(planGemmaContextMemory(131_072)).toMatchObject({
    largestCacheBufferBytes: 268_435_456,
    kvCacheBytes: 1_623_195_648,
  });
});

test("rejects model and adapter capacity overruns before allocation", () => {
  expect(() => planGemmaContextMemory(131_073)).toThrow(/between 1 and 131072/);
  expect(() => assertGemmaContextSupported(131_072, {
    maxBufferSize: 268_435_456,
    maxStorageBufferBindingSize: 268_435_455,
  })).toThrow(/exceeding device limit 268435455/);
  expect(assertGemmaContextSupported(131_072, {
    maxBufferSize: 1_073_741_824,
    maxStorageBufferBindingSize: 1_073_741_824,
  }).fullPhysicalCapacity).toBe(131_072);
});

test("reserves one output evaluation inside the context capacity", () => {
  expect(availableGemmaOutputTokens(1, 512)).toBe(512);
  expect(availableGemmaOutputTokens(511, 512)).toBe(2);
  expect(availableGemmaOutputTokens(512, 512)).toBe(1);
  expect(availableGemmaOutputTokens(513, 512)).toBe(0);
  expect(availableGemmaOutputTokens(513, 8_192)).toBe(7_680);
  expect(availableGemmaOutputTokens(8_191, 8_192)).toBe(2);
  expect(availableGemmaOutputTokens(8_192, 8_192)).toBe(1);
  expect(availableGemmaOutputTokens(8_193, 8_192)).toBe(0);
  expect(availableGemmaOutputTokens(32_767, 32_768)).toBe(2);
  expect(availableGemmaOutputTokens(32_768, 32_768)).toBe(1);
  expect(availableGemmaOutputTokens(32_769, 32_768)).toBe(0);
  expect(availableGemmaOutputTokens(131_071, 131_072)).toBe(2);
  expect(availableGemmaOutputTokens(131_072, 131_072)).toBe(1);
  expect(availableGemmaOutputTokens(131_073, 131_072)).toBe(0);
});