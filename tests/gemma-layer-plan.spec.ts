import { expect, test } from "@playwright/test";
import type { CachedTensorDescriptor } from "../src/model/cached-safetensors";
import {
  createGemmaLayerPlan,
  createGemmaLayerPlans,
  gemmaLayerTensorContracts,
} from "../src/model/gemma-layer-plan";
import { createGemmaDecodeStackSchedule } from "../src/webgpu/decode-stack";

test("derives all four exact Gemma layer execution profiles", () => {
  const descriptors = createAllDescriptors();
  const plans = createGemmaLayerPlans(descriptors);

  expect(plans).toHaveLength(35);
  expect(countProfiles(plans.map(({ profile }) => profile))).toEqual({
    "sliding-int4": 12,
    "full-int4": 3,
    "sliding-int2": 16,
    "full-int2": 4,
  });
  expect(plans[0].attention).toMatchObject({
    type: "sliding_attention",
    headDim: 256,
    rotaryDimensions: 256,
    slidingWindow: 512,
    qOutFeatures: 2048,
    kvOutFeatures: 256,
  });
  expect(plans[4].attention).toMatchObject({
    type: "full_attention",
    headDim: 512,
    rotaryDimensions: 128,
    slidingWindow: null,
    qOutFeatures: 4096,
    kvOutFeatures: 512,
  });
  expect(plans[14].attention.kNorm?.shape).toEqual([512]);
  expect(plans[15].attention.kNorm).toBeNull();
  expect(plans[13].attention).toMatchObject({
    isKvShared: false,
    kvSourceLayer: 13,
  });
  expect(plans[14].attention).toMatchObject({
    isKvShared: false,
    kvSourceLayer: 14,
  });
  expect(plans[15].attention).toMatchObject({
    isKvShared: true,
    kvSourceLayer: 13,
  });
  expect(plans[19].attention).toMatchObject({
    isKvShared: true,
    kvSourceLayer: 14,
  });
  expect(plans[34].attention).toMatchObject({
    isKvShared: true,
    kvSourceLayer: 14,
  });
  expect(plans[14].mlp).toMatchObject({ bits: 4, intermediateSize: 6144 });
  expect(plans[15].mlp).toMatchObject({ bits: 2, intermediateSize: 12288 });
  expect(plans[15].mlp.gate.weight.shape).toEqual([12288, 384]);
  expect(plans[15].mlp.down.weight.shape).toEqual([1536, 3072]);
});

test("derives the accelerated 35-layer decode schedule", () => {
  const plans = createGemmaLayerPlans(createAllDescriptors());
  const schedule = createGemmaDecodeStackSchedule(plans);

  expect(schedule).toHaveLength(35);
  expect(schedule[0]).toEqual({
    layerIndex: 0,
    mode: "initial",
    kvSourceLayer: 0,
    dispatches: 10,
  });
  expect(schedule.slice(1, 15).every(({ mode, dispatches }) =>
    mode === "owned-kv" && dispatches === 9)).toBe(true);
  expect(schedule.slice(15).every(({ mode, dispatches }) =>
    mode === "shared-kv" && dispatches === 7)).toBe(true);
  expect(schedule[15].kvSourceLayer).toBe(13);
  expect(schedule[19].kvSourceLayer).toBe(14);
  expect(schedule.reduce((total, { dispatches }) => total + dispatches, 0)).toBe(276);
});

test("covers every required model tensor without fixture intermediates", () => {
  const contracts = Array.from({ length: 35 }, (_, layerIndex) =>
    gemmaLayerTensorContracts(layerIndex));
  const names = contracts.flat().map(({ name }) => name);

  expect(contracts.slice(0, 15).every((layer) => layer.length === 44)).toBe(true);
  expect(contracts.slice(15).every((layer) => layer.length === 35)).toBe(true);
  expect(names).toHaveLength(1360);
  expect(new Set(names).size).toBe(names.length);
  expect(names.some((name) => name.includes("gateGeluLut"))).toBe(false);
});

test("rejects missing and structurally mismatched layer tensors", () => {
  const descriptors = createLayerDescriptors(15);
  const gateWeightName = "model.language_model.layers.15.mlp.gate_proj.weight";
  const gateWeight = descriptors.get(gateWeightName);
  if (!gateWeight) throw new Error("Test descriptor is missing gate weight");

  descriptors.set(gateWeightName, { ...gateWeight, shape: [6144, 768] });
  expect(() => createGemmaLayerPlan(descriptors, 15)).toThrow(/tensor contract mismatch/);

  descriptors.set(gateWeightName, gateWeight);
  descriptors.delete("model.language_model.layers.15.layer_scalar");
  expect(() => createGemmaLayerPlan(descriptors, 15)).toThrow(/is missing tensor/);
});

test("rejects layer indices outside the pinned 35-layer model", () => {
  expect(() => gemmaLayerTensorContracts(-1)).toThrow(/0 through 34/);
  expect(() => gemmaLayerTensorContracts(35)).toThrow(/0 through 34/);
});

function createAllDescriptors(): Map<string, CachedTensorDescriptor> {
  const descriptors = new Map<string, CachedTensorDescriptor>();
  for (let layerIndex = 0; layerIndex < 35; layerIndex += 1) {
    for (const [name, descriptor] of createLayerDescriptors(layerIndex)) {
      descriptors.set(name, descriptor);
    }
  }
  return descriptors;
}

function createLayerDescriptors(layerIndex: number): Map<string, CachedTensorDescriptor> {
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
  return descriptors;
}

function countProfiles(profiles: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const profile of profiles) counts[profile] = (counts[profile] ?? 0) + 1;
  return counts;
}