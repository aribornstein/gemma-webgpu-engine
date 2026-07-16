import { expect, test } from "@playwright/test";
import {
  canRetainDecodeKvPrefix,
  resolveDecodeKvCacheAllocation,
} from "../src/webgpu/decode-kv-cache";

test("uses a fixed circular window for sliding attention", () => {
  for (const cacheCapacity of [512, 513, 2_048, 8_192, 32_768, 131_072]) {
    expect(resolveDecodeKvCacheAllocation({
      keyLength: 1,
      cacheCapacity,
      window: 512,
    })).toEqual({ capacity: 512, mode: "circular" });
  }
});

test("uses the configured logical capacity for full attention", () => {
  for (const cacheCapacity of [2_048, 8_192, 32_768, 131_072]) {
    expect(resolveDecodeKvCacheAllocation({
      keyLength: 1,
      cacheCapacity,
      window: 0,
    })).toEqual({ capacity: cacheCapacity, mode: "linear" });
  }
});

test("retains only a safe rollback after a circular cache wraps", () => {
  expect(canRetainDecodeKvPrefix("circular", 512, 32_768, 32_768)).toBe(true);
  expect(canRetainDecodeKvPrefix("circular", 512, 32_768, 32_767)).toBe(true);
  expect(canRetainDecodeKvPrefix("circular", 512, 32_768, 32_766)).toBe(false);
  expect(canRetainDecodeKvPrefix("linear", 32_768, 32_768, 1)).toBe(true);
});