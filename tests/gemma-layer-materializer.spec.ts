import { expect, test } from "@playwright/test";
import type { CachedTensorPayload } from "../src/model/cached-safetensors";
import {
  bfloat16ToFloat32,
  createPleNormWeights,
  float32LittleEndian,
  materializeGemmaLayerWeights,
  packedUint32,
} from "../src/model/gemma-layer-materializer";
import { createGemmaLayerPlan, gemmaLayerTensorContracts } from "../src/model/gemma-layer-plan";
import type { LoadedGemmaLayerWeights } from "../src/model/gemma-layer-weights";

test("decodes canonical little-endian tensor storage", () => {
  expect(Array.from(packedUint32(payload("codes", "U8", [4], [0x12, 0x34, 0x56, 0x78]))))
    .toEqual([0x78563412]);
  expect(Array.from(packedUint32(payload("signed", "I8", [4], [0x80, 0xff, 0x00, 0x7f]))))
    .toEqual([0xff807f00]);
  expect(Array.from(float32LittleEndian(payload(
    "scale",
    "F32",
    [2],
    bytesFromFloat32([1.5, -2.25]),
  )))).toEqual([1.5, -2.25]);
  expect(Array.from(bfloat16ToFloat32(payload(
    "norm",
    "BF16",
    [3],
    [0x80, 0x3f, 0x80, 0xbf, 0x00, 0x40],
  )))).toEqual([1, -1, 2]);
});

test("materializes the exact layer-0 kernel resource layouts", () => {
  const weights = createWeights(0);
  const materialized = materializeGemmaLayerWeights(weights);
  const nextInputNorm = Float32Array.from({ length: 1536 }, () => 7);
  const pleNormWeights = createPleNormWeights(materialized, nextInputNorm);

  expect(materialized.profile).toBe("sliding-int4");
  expect(materialized.qkv.packedWeights.length).toBe((2048 + 256 + 256) * 192);
  expect(materialized.qkv.rowScales.length).toBe(2048 + 256 + 256);
  expect(Array.from(materialized.qkv.outputScales)).toEqual([0.25, 0.25, 0.25]);
  expect(materialized.norms.oProjectionFused.length).toBe(3072);
  expect(Array.from(materialized.norms.oProjectionFused.slice(0, 2))).toEqual([2, 2]);
  expect(Array.from(materialized.norms.oProjectionFused.slice(1536, 1538))).toEqual([3, 3]);
  expect(materialized.norms.postFeedforward[0]).toBe(4);
  expect(materialized.layerScalar).toBe(0.5);
  expect(pleNormWeights.length).toBe(3073);
  expect(pleNormWeights[0]).toBe(5);
  expect(pleNormWeights[1536]).toBe(7);
  expect(pleNormWeights[3072]).toBe(0.5);
});

test("rejects inconsistent fused QKV input scales", () => {
  const weights = createWeights(0);
  const name = "model.language_model.layers.0.self_attn.k_proj.input_activation_scale";
  weights.tensors.set(name, payload(name, "F32", [], bytesFromFloat32([0.75])));
  expect(() => materializeGemmaLayerWeights(weights)).toThrow(/input scales do not match/);
});

test("materializes shared layers with Q-only attention weights", () => {
  const materialized = materializeGemmaLayerWeights(createWeights(15));

  expect(materialized.qkv.packedWeights.length).toBe(2048 * 192);
  expect(materialized.qkv.rowScales.length).toBe(2048);
  expect(Array.from(materialized.qkv.outputScales)).toEqual([0.25, 0, 0]);
  expect(materialized.norms.k).toBeNull();
});

function createWeights(layerIndex: number): LoadedGemmaLayerWeights & {
  tensors: Map<string, CachedTensorPayload>;
} {
  const tensors = new Map<string, CachedTensorPayload>();
  let begin = 375400;
  for (const contract of gemmaLayerTensorContracts(layerIndex)) {
    const bytesPerElement = contract.dtype === "F32" ? 4 : contract.dtype === "BF16" ? 2 : 1;
    const elements = contract.shape.reduce((product, dimension) => product * dimension, 1);
    const byteLength = elements * bytesPerElement;
    let bytes = new Uint8Array(byteLength);
    if (contract.dtype === "F32") {
      const value = contract.name.endsWith("input_activation_scale")
        ? 0.5
        : contract.name.endsWith("output_activation_scale") ? 0.25 : 0;
      bytes = new Uint8Array(bytesFromFloat32(Array.from({ length: elements }, () => value)));
    } else if (contract.dtype === "BF16") {
      const value = contract.name.endsWith("input_layernorm.weight") ? 1
        : contract.name.endsWith("post_attention_layernorm.weight") ? 2
        : contract.name.endsWith("pre_feedforward_layernorm.weight") ? 3
        : contract.name.endsWith("post_feedforward_layernorm.weight") ? 4
        : contract.name.endsWith("post_per_layer_input_norm.weight") ? 5
        : contract.name.endsWith("layer_scalar") ? 0.5
        : 1;
      bytes = bytesFromBfloat16(value, elements);
    }
    tensors.set(contract.name, {
      ...contract,
      begin,
      end: begin + byteLength,
      byteLength,
      bytes,
      sha256: "0".repeat(64),
    });
    begin += byteLength;
  }
  const descriptors = new Map(Array.from(tensors, ([name, tensor]) => [name, tensor]));
  const plan = createGemmaLayerPlan(descriptors, layerIndex);
  return {
    plan,
    tensors,
    tensorHashes: new Map(Array.from(tensors, ([name, tensor]) => [name, tensor.sha256])),
    bytesLoaded: plan.tensorBytes,
  };
}

function payload(
  name: string,
  dtype: string,
  shape: readonly number[],
  values: ArrayLike<number>,
): CachedTensorPayload {
  const bytes = Uint8Array.from(values);
  return {
    name,
    dtype,
    shape,
    begin: 0,
    end: bytes.byteLength,
    byteLength: bytes.byteLength,
    bytes,
    sha256: "0".repeat(64),
  };
}

function bytesFromFloat32(values: readonly number[]): number[] {
  const array = new Float32Array(values);
  return Array.from(new Uint8Array(array.buffer));
}

function bytesFromBfloat16(value: number, count: number): Uint8Array<ArrayBuffer> {
  const valueBits = new Uint32Array(new Float32Array([value]).buffer)[0] >>> 16;
  const bytes = new Uint8Array(count * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < count; index += 1) view.setUint16(index * 2, valueBits, true);
  return bytes;
}