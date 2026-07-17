import {
  GemmaVisionWeightCache,
  loadGemmaVisionLayerWeights,
  type GemmaVisionTensorSource,
} from "../model/gemma-vision-weights";
import {
  createGemmaVisionLayerResources,
  destroyGemmaVisionLayerResources,
  encodeGemmaVisionLayer,
} from "./vision-layer";

export const GEMMA_VISION_LAYER_COUNT = 16;
const HIDDEN_SIZE = 768;

export interface GemmaVisionEncoderProgress {
  layerIndex: number;
  completedLayers: number;
  layerSourceBytes: number;
}

export interface GemmaVisionEncoderTiming {
  weightLoadMilliseconds: number;
  resourceSetupMilliseconds: number;
  executionMilliseconds: number;
}

export interface GemmaVisionEncoderResult {
  output: GPUBuffer;
  layers: number;
  sourceBytes: number;
  elapsedMilliseconds: number;
  timing: GemmaVisionEncoderTiming;
}

export async function runGemmaVisionEncoder(
  device: GPUDevice,
  source: GemmaVisionTensorSource,
  hidden: GPUBuffer,
  rows: number,
  positions: Int32Array,
  onProgress?: (progress: GemmaVisionEncoderProgress) => void,
  signal?: AbortSignal,
  weightCache?: GemmaVisionWeightCache,
): Promise<GemmaVisionEncoderResult> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 2520 ||
      positions.length < rows * 2 || hidden.size < rows * HIDDEN_SIZE * 4) {
    throw new Error("Gemma vision encoder input geometry is invalid");
  }
  const started = performance.now();
  let sourceBytes = 0;
  let weightLoadMilliseconds = 0;
  let resourceSetupMilliseconds = 0;
  let executionMilliseconds = 0;
  for (let layerIndex = 0; layerIndex < GEMMA_VISION_LAYER_COUNT; layerIndex += 1) {
    throwIfAborted(signal);
    const weightLoadStarted = performance.now();
    const weights = await (weightCache
      ? weightCache.loadLayer(source, layerIndex)
      : loadGemmaVisionLayerWeights(source, layerIndex));
    weightLoadMilliseconds += performance.now() - weightLoadStarted;
    throwIfAborted(signal);
    sourceBytes += weights.sourceBytes;
    const resourceSetupStarted = performance.now();
    const resources = await createGemmaVisionLayerResources(
      device,
      hidden,
      rows,
      positions,
      weights,
    );
    resourceSetupMilliseconds += performance.now() - resourceSetupStarted;
    try {
      throwIfAborted(signal);
      const executionStarted = performance.now();
      const encoder = device.createCommandEncoder({
        label: `Gemma vision encoder layer ${layerIndex}`,
      });
      encodeGemmaVisionLayer(encoder, resources);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      executionMilliseconds += performance.now() - executionStarted;
      throwIfAborted(signal);
    } finally {
      destroyGemmaVisionLayerResources(resources);
    }
    onProgress?.({
      layerIndex,
      completedLayers: layerIndex + 1,
      layerSourceBytes: weights.sourceBytes,
    });
  }
  return {
    output: hidden,
    layers: GEMMA_VISION_LAYER_COUNT,
    sourceBytes,
    elapsedMilliseconds: performance.now() - started,
    timing: {
      weightLoadMilliseconds,
      resourceSetupMilliseconds,
      executionMilliseconds,
    },
  };
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Gemma vision encoding was cancelled", "AbortError");
}