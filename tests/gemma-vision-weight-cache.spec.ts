import { expect, test } from "@playwright/test";
import type { CachedTensorPayload } from "../src/model/cached-safetensors";
import {
  GEMMA_VISION_PROJECTOR_WEIGHT,
  GemmaVisionWeightCache,
  type GemmaVisionTensorSource,
} from "../src/model/gemma-vision-weights";

test("reuses immutable materialized vision weights until cleared", async () => {
  const byteLength = 1536 * 768 * 4;
  const descriptor = {
    name: GEMMA_VISION_PROJECTOR_WEIGHT,
    dtype: "F32",
    shape: [1536, 768],
    begin: 0,
    end: byteLength,
    byteLength,
  } as const;
  const payload: CachedTensorPayload = {
    ...descriptor,
    bytes: new Uint8Array(byteLength),
    sha256: "0".repeat(64),
  };
  let reads = 0;
  const source: GemmaVisionTensorSource = {
    descriptors: new Map([[descriptor.name, descriptor]]),
    async readTensors(names) {
      reads += 1;
      expect(names).toEqual([GEMMA_VISION_PROJECTOR_WEIGHT]);
      return new Map([[payload.name, payload]]);
    },
  };
  const cache = new GemmaVisionWeightCache();

  const first = await cache.loadProjector(source);
  const second = await cache.loadProjector(source);
  expect(second).toBe(first);
  expect(reads).toBe(1);

  cache.clear();
  const afterClear = await cache.loadProjector(source);
  expect(afterClear).not.toBe(first);
  expect(reads).toBe(2);
});