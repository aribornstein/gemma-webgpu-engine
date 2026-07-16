import type {
  CachedTensorDescriptor,
  CachedTensorPayload,
  ReadonlySafetensorsCache,
} from "./cached-safetensors";
import { createGemmaLayerPlan } from "./gemma-layer-plan";
import type { GemmaLayerPlan } from "./gemma-layer-plan";

export interface GemmaLayerTensorSource {
  readonly descriptors: ReadonlyMap<string, CachedTensorDescriptor>;
  readTensors(names: readonly string[]): Promise<Map<string, CachedTensorPayload>>;
}

export interface LoadedGemmaLayerWeights {
  plan: GemmaLayerPlan;
  tensors: ReadonlyMap<string, CachedTensorPayload>;
  tensorHashes: ReadonlyMap<string, string>;
  bytesLoaded: number;
}

export async function loadGemmaLayerWeights(
  source: GemmaLayerTensorSource | ReadonlySafetensorsCache,
  layerIndex: number,
): Promise<LoadedGemmaLayerWeights> {
  const plan = createGemmaLayerPlan(source.descriptors, layerIndex);
  const tensors = await source.readTensors(plan.tensorNames);
  let bytesLoaded = 0;
  const tensorHashes = new Map<string, string>();
  for (const name of plan.tensorNames) {
    const expected = source.descriptors.get(name);
    const tensor = tensors.get(name);
    if (!expected || !tensor) {
      throw new Error(`Gemma layer ${layerIndex} weight load omitted tensor ${name}`);
    }
    if (
      tensor.name !== expected.name ||
      tensor.dtype !== expected.dtype ||
      tensor.byteLength !== expected.byteLength ||
      tensor.bytes.byteLength !== expected.byteLength ||
      tensor.sha256.length !== 64
    ) {
      throw new Error(`Gemma layer ${layerIndex} loaded payload mismatch for ${name}`);
    }
    bytesLoaded += tensor.byteLength;
    tensorHashes.set(name, tensor.sha256);
  }
  if (tensors.size !== plan.tensorNames.length) {
    throw new Error(
      `Gemma layer ${layerIndex} weight load returned ${tensors.size} tensors; ` +
      `expected ${plan.tensorNames.length}`,
    );
  }
  return { plan, tensors, tensorHashes, bytesLoaded };
}

export function requiredLayerTensor(
  weights: LoadedGemmaLayerWeights,
  name: string,
): CachedTensorPayload {
  const tensor = weights.tensors.get(name);
  if (!tensor) throw new Error(`Loaded Gemma layer is missing tensor ${name}`);
  return tensor;
}