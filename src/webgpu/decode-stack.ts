import type { DecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import {
  createPleNormWeights,
  materializeGemmaLayerWeights,
  type MaterializedGemmaLayer,
} from "../model/gemma-layer-materializer";
import {
  createGemmaLayerPlans,
  type GemmaLayerPlan,
} from "../model/gemma-layer-plan";
import {
  loadGemmaLayerWeights,
  type GemmaLayerTensorSource,
} from "../model/gemma-layer-weights";
import {
  commitDecodeAttentionBlockCache,
  createGemmaDecodeAttentionBlockResources,
  createGemmaDecodeSharedKvAttentionBlockResources,
  destroyDecodeAttentionBlockResources,
  updateDecodeAttentionBlockToken,
} from "./decode-attention-block";
import type { DecodeKvCache } from "./decode-kv-cache";
import {
  encodeGemmaDecodeLayer,
  gemmaDecodeLayerDispatchCount,
  getGemmaDecodeLayerPipelines,
  type GemmaDecodeLayerPipelines,
  type GemmaDecodeLayerResources,
} from "./decode-layer";
import {
  createDecodeMlpPleBlockResources,
  destroyDecodeMlpPleBlockResources,
} from "./decode-mlp-ple-block";

const LAYER_COUNT = 35;
const Q_HEADS: 8 = 8;
const KV_HEADS: 1 = 1;

export interface GemmaDecodeRotaryRow {
  cosine: Float32Array;
  sine: Float32Array;
}

export interface GemmaDecodeCachePrefix {
  keys: Float32Array;
  values: Float32Array;
}

export interface GemmaDecodeStackRuntime {
  hidden: Float32Array;
  hiddenBuffer?: GPUBuffer;
  perLayerInputsBuffer?: GPUBuffer;
  keyLength: number;
  queryOffset: number;
  cacheCapacity: number;
  slidingRotary: GemmaDecodeRotaryRow;
  fullRotary: GemmaDecodeRotaryRow;
  cachePrefixes?: ReadonlyMap<number, GemmaDecodeCachePrefix>;
  finalInputNorm: Float32Array;
  finalInputScale: number;
}

export interface GemmaDecodeStackResources {
  pipelines: readonly GemmaDecodeLayerPipelines[];
  layers: readonly GemmaDecodeLayerResources[];
  ownerCaches: ReadonlyMap<number, DecodeKvCache>;
  dispatchesPerToken: number;
  hidden: GPUBuffer;
  finalInput: GPUBuffer;
  finalSum: GPUBuffer;
}

export interface GemmaDecodeStackScheduleEntry {
  layerIndex: number;
  mode: "initial" | "owned-kv" | "shared-kv";
  kvSourceLayer: number;
  dispatches: 7 | 9 | 10;
}

export function createGemmaDecodeStackSchedule(
  plans: readonly GemmaLayerPlan[],
): GemmaDecodeStackScheduleEntry[] {
  if (plans.length !== LAYER_COUNT) {
    throw new Error(`Gemma decode stack requires ${LAYER_COUNT} plans`);
  }
  const ownerLayers = new Set<number>();
  return plans.map((plan, layerIndex) => {
    if (plan.layerIndex !== layerIndex) {
      throw new Error(`Gemma decode stack plan ${layerIndex} is out of order`);
    }
    if (!plan.attention.isKvShared) {
      ownerLayers.add(layerIndex);
      return {
        layerIndex,
        mode: layerIndex === 0 ? "initial" : "owned-kv",
        kvSourceLayer: layerIndex,
        dispatches: layerIndex === 0 ? 10 : 9,
      };
    }
    if (!ownerLayers.has(plan.attention.kvSourceLayer)) {
      throw new Error(
        `Gemma layer ${layerIndex} references unavailable K/V source ${plan.attention.kvSourceLayer}`,
      );
    }
    const source = plans[plan.attention.kvSourceLayer];
    if (source.attention.type !== plan.attention.type) {
      throw new Error(`Gemma layer ${layerIndex} K/V source has incompatible attention geometry`);
    }
    return {
      layerIndex,
      mode: "shared-kv",
      kvSourceLayer: plan.attention.kvSourceLayer,
      dispatches: 7,
    };
  });
}

export async function createGemmaDecodeStackResources(
  device: GPUDevice,
  plans: readonly GemmaLayerPlan[],
  materializedLayers: readonly MaterializedGemmaLayer[],
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeStackRuntime,
): Promise<GemmaDecodeStackResources> {
  validateStackInputs(plans, materializedLayers, runtime);
  return buildGemmaDecodeStackResources(
    device,
    plans,
    (layerIndex) => Promise.resolve(materializedLayers[layerIndex]),
    fixture,
    runtime,
  );
}

export async function loadGemmaDecodeStackResources(
  device: GPUDevice,
  source: GemmaLayerTensorSource,
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeStackRuntime,
): Promise<GemmaDecodeStackResources> {
  const plans = createGemmaLayerPlans(source.descriptors);
  validateStackRuntime(runtime);
  return buildGemmaDecodeStackResources(
    device,
    plans,
    async (layerIndex) => materializeGemmaLayerWeights(
      await loadGemmaLayerWeights(source, layerIndex),
    ),
    fixture,
    runtime,
  );
}

async function buildGemmaDecodeStackResources(
  device: GPUDevice,
  plans: readonly GemmaLayerPlan[],
  loadLayer: (layerIndex: number) => Promise<MaterializedGemmaLayer>,
  fixture: DecodeMlpPleFixture,
  runtime: GemmaDecodeStackRuntime,
): Promise<GemmaDecodeStackResources> {
  createGemmaDecodeStackSchedule(plans);
  const layers: GemmaDecodeLayerResources[] = [];
  const stackPipelines: GemmaDecodeLayerPipelines[] = [];
  const ownerCaches = new Map<number, DecodeKvCache>();
  let layer = await loadLayer(0);
  let nextLayerPromise: Promise<MaterializedGemmaLayer> | null = loadLayer(1);

  try {
    for (let layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex += 1) {
      const plan = plans[layerIndex];
      validateMaterializedLayer(plan, layer);
      const nextLayer = nextLayerPromise ? await nextLayerPromise : undefined;
      if (nextLayer) validateMaterializedLayer(plans[layerIndex + 1], nextLayer);
      const pipelines = await getGemmaDecodeLayerPipelines(device, plan.profile);
      const rotary = plan.attention.type === "full_attention"
        ? runtime.fullRotary
        : runtime.slidingRotary;
      const previous = layers.at(-1);
      const activations = previous
        ? {
            input: previous.mlp.nextInput,
            inputSum: previous.mlp.nextSum,
            hidden: previous.mlp.hidden,
          }
        : undefined;
      const commonRuntime = {
        hidden: runtime.hidden,
        hiddenBuffer: layerIndex === 0 ? runtime.hiddenBuffer : undefined,
        cosine: rotary.cosine,
        sine: rotary.sine,
        keyLength: runtime.keyLength,
        cacheCapacity: runtime.cacheCapacity,
        queryOffset: runtime.queryOffset,
        qHeads: Q_HEADS,
        kvHeads: KV_HEADS,
        window: plan.attention.slidingWindow ?? 0,
      };
      let attention;
      if (plan.attention.isKvShared) {
        const sourceCache = ownerCaches.get(plan.attention.kvSourceLayer);
        if (!sourceCache) {
          throw new Error(
            `Gemma layer ${layerIndex} cannot find K/V source layer ${plan.attention.kvSourceLayer}`,
          );
        }
        attention = createGemmaDecodeSharedKvAttentionBlockResources(
          device,
          pipelines.attention,
          layer,
          { ...commonRuntime, sourceCache },
          activations,
        );
      } else {
        const prefix = runtime.cachePrefixes?.get(layerIndex);
        const keyCache = prefix?.keys ?? new Float32Array(0);
        const valueCache = prefix?.values ?? new Float32Array(0);
        validateCachePrefix(plan, runtime.queryOffset, keyCache, valueCache);
        attention = createGemmaDecodeAttentionBlockResources(
          device,
          pipelines.attention,
          layer,
          { ...commonRuntime, keyCache, valueCache },
          activations,
        );
        ownerCaches.set(layerIndex, attention.cache);
      }

      let mlp;
      try {
        mlp = createDecodeMlpPleBlockResources(
          device,
          pipelines.mlp,
          fixture,
          {
            preMlpInput: attention.ffnInputBuffer,
            preMlpSum: attention.ffnInputSumBuffer,
            hidden: attention.hiddenBuffer,
            pleInput: runtime.perLayerInputsBuffer,
            pleInputOffset: runtime.perLayerInputsBuffer ? layerIndex * 256 * 4 : undefined,
          },
          {
            layer,
            pleNormWeights: createPleNormWeights(
              layer,
              nextLayer?.norms.input ?? runtime.finalInputNorm,
            ),
            nextInputScale: nextLayer?.qkv.inputScale ?? runtime.finalInputScale,
          },
        );
      } catch (error) {
        destroyDecodeAttentionBlockResources(attention);
        ownerCaches.delete(layerIndex);
        throw error;
      }
      stackPipelines.push(pipelines);
      layers.push({ attention, mlp });
      layer = nextLayer ?? layer;
      nextLayerPromise = layerIndex + 2 < LAYER_COUNT
        ? loadLayer(layerIndex + 2)
        : null;
    }
  } catch (error) {
    destroyLayers(layers);
    throw error;
  }

  const finalLayer = layers[LAYER_COUNT - 1];
  return {
    pipelines: stackPipelines,
    layers,
    ownerCaches,
    dispatchesPerToken: layers.reduce(
      (total, resources) => total + gemmaDecodeLayerDispatchCount(resources),
      0,
    ),
    hidden: finalLayer.mlp.hidden,
    finalInput: finalLayer.mlp.nextInput,
    finalSum: finalLayer.mlp.nextSum,
  };
}

export function encodeGemmaDecodeStack(
  encoder: GPUCommandEncoder,
  resources: GemmaDecodeStackResources,
): void {
  if (resources.layers.length !== LAYER_COUNT) {
    throw new Error(`Gemma decode stack requires ${LAYER_COUNT} layers`);
  }
  for (let layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex += 1) {
    encodeGemmaDecodeLayer(
      encoder,
      resources.pipelines[layerIndex],
      resources.layers[layerIndex],
    );
  }
}

export function updateGemmaDecodeStackToken(
  device: GPUDevice,
  resources: GemmaDecodeStackResources,
  position: number,
  slidingRotary: GemmaDecodeRotaryRow,
  fullRotary: GemmaDecodeRotaryRow,
): void {
  for (const [layerIndex, cache] of resources.ownerCaches) {
    if (cache.length !== position) {
      throw new Error(
        `Gemma owner cache ${layerIndex} length ${cache.length} does not match position ${position}`,
      );
    }
  }
  for (let layerIndex = 0; layerIndex < resources.layers.length; layerIndex += 1) {
    const rotary = resources.pipelines[layerIndex].attention.profile.startsWith("full")
      ? fullRotary
      : slidingRotary;
    updateDecodeAttentionBlockToken(
      device,
      resources.layers[layerIndex].attention,
      position,
      rotary.cosine,
      rotary.sine,
    );
  }
}

export async function submitGemmaDecodeStack(
  device: GPUDevice,
  resources: GemmaDecodeStackResources,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "Gemma 35-layer decode stack" });
  encodeGemmaDecodeStack(encoder, resources);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  commitGemmaDecodeStackCaches(resources);
}

export function commitGemmaDecodeStackCaches(
  resources: GemmaDecodeStackResources,
): void {
  for (const layer of resources.layers) {
    commitDecodeAttentionBlockCache(layer.attention);
  }
}

export function destroyGemmaDecodeStackResources(
  resources: GemmaDecodeStackResources,
): void {
  destroyLayers(resources.layers);
}

function destroyLayers(layers: readonly GemmaDecodeLayerResources[]): void {
  for (const layer of layers.toReversed()) {
    destroyDecodeMlpPleBlockResources(layer.mlp);
    destroyDecodeAttentionBlockResources(layer.attention);
  }
}

function validateStackInputs(
  plans: readonly GemmaLayerPlan[],
  layers: readonly MaterializedGemmaLayer[],
  runtime: GemmaDecodeStackRuntime,
): void {
  if (plans.length !== LAYER_COUNT || layers.length !== LAYER_COUNT) {
    throw new Error(`Gemma decode stack requires ${LAYER_COUNT} plans and materialized layers`);
  }
  validateStackRuntime(runtime);
  for (let layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex += 1) {
    validateMaterializedLayer(plans[layerIndex], layers[layerIndex]);
  }
}

function validateStackRuntime(runtime: GemmaDecodeStackRuntime): void {
  if (runtime.hidden.length !== 1536 || runtime.finalInputNorm.length !== 1536) {
    throw new Error("Gemma decode stack hidden and final norm must contain 1536 values");
  }
  if (!Number.isInteger(runtime.queryOffset) || runtime.queryOffset < 0 ||
      !Number.isInteger(runtime.keyLength) || runtime.keyLength !== runtime.queryOffset + 1 ||
      !Number.isInteger(runtime.cacheCapacity) || runtime.cacheCapacity < runtime.keyLength) {
    throw new Error("Gemma decode stack cache geometry is invalid");
  }
  if (runtime.slidingRotary.cosine.length !== 128 ||
      runtime.slidingRotary.sine.length !== 128 ||
      runtime.fullRotary.cosine.length !== 256 ||
      runtime.fullRotary.sine.length !== 256) {
    throw new Error("Gemma decode stack rotary rows do not match model geometry");
  }
}

function validateMaterializedLayer(
  plan: GemmaLayerPlan,
  layer: MaterializedGemmaLayer,
): void {
  if (plan.layerIndex !== layer.layerIndex || plan.profile !== layer.profile) {
    throw new Error(`Gemma decode stack layer ${plan.layerIndex} is out of order`);
  }
}

function validateCachePrefix(
  plan: GemmaLayerPlan,
  queryOffset: number,
  keys: Float32Array,
  values: Float32Array,
): void {
  const expectedElements = queryOffset * plan.attention.kvOutFeatures;
  if (keys.length !== expectedElements || values.length !== expectedElements) {
    throw new Error(
      `Gemma layer ${plan.layerIndex} cache prefix must contain ${expectedElements} elements`,
    );
  }
}