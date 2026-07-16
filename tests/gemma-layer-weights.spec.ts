import { expect, test } from "@playwright/test";
import type {
  CachedTensorDescriptor,
  CachedTensorPayload,
} from "../src/model/cached-safetensors";
import { gemmaLayerTensorContracts } from "../src/model/gemma-layer-plan";
import {
  loadGemmaLayerWeights,
  type GemmaLayerTensorSource,
} from "../src/model/gemma-layer-weights";

test("loads one canonical layer batch with retained tensor hashes", async () => {
  const source = createLayerSource(15);
  const loaded = await loadGemmaLayerWeights(source, 15);

  expect(loaded.plan.profile).toBe("sliding-int2");
  expect(loaded.tensors.size).toBe(35);
  expect(loaded.tensorHashes.size).toBe(35);
  expect(loaded.bytesLoaded).toBe(loaded.plan.tensorBytes);
  expect(source.readCalls).toEqual([loaded.plan.tensorNames]);
});

test("rejects a cache batch that omits a planned tensor", async () => {
  const source = createLayerSource(0);
  source.omit = "model.language_model.layers.0.layer_scalar";

  await expect(loadGemmaLayerWeights(source, 0)).rejects.toThrow(/omitted tensor/);
});

interface TestTensorSource extends GemmaLayerTensorSource {
  readCalls: readonly string[][];
  omit: string | null;
}

function createLayerSource(layerIndex: number): TestTensorSource {
  const descriptors = new Map<string, CachedTensorDescriptor>();
  let begin = 375400;
  for (const tensor of gemmaLayerTensorContracts(layerIndex)) {
    const byteLength = tensor.shape.reduce((product, dimension) => product * dimension, 1) *
      (tensor.dtype === "F32" ? 4 : tensor.dtype === "BF16" ? 2 : 1);
    descriptors.set(tensor.name, {
      ...tensor,
      begin,
      end: begin + byteLength,
      byteLength,
    });
    begin += byteLength;
  }

  const source: TestTensorSource = {
    descriptors,
    readCalls: [],
    omit: null,
    async readTensors(names): Promise<Map<string, CachedTensorPayload>> {
      source.readCalls = [...source.readCalls, [...names]];
      const payloads = new Map<string, CachedTensorPayload>();
      for (const name of names) {
        if (name === source.omit) continue;
        const descriptor = descriptors.get(name);
        if (!descriptor) throw new Error(`Test source is missing ${name}`);
        payloads.set(name, {
          ...descriptor,
          bytes: new Uint8Array(descriptor.byteLength),
          sha256: "0".repeat(64),
        });
      }
      return payloads;
    },
  };
  return source;
}