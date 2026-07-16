import { loadGemmaVisionLayerWeights } from "../model/gemma-vision-weights";
import { PinnedSafetensorsSource } from "../model/pinned-safetensors";
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

export interface GemmaVisionEncoderResult {
  output: GPUBuffer;
  layers: number;
  sourceBytes: number;
  elapsedMilliseconds: number;
}

export async function runGemmaVisionEncoder(
  device: GPUDevice,
  source: PinnedSafetensorsSource,
  hidden: GPUBuffer,
  rows: number,
  positions: Int32Array,
  onProgress?: (progress: GemmaVisionEncoderProgress) => void,
): Promise<GemmaVisionEncoderResult> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 2520 ||
      positions.length < rows * 2 || hidden.size < rows * HIDDEN_SIZE * 4) {
    throw new Error("Gemma vision encoder input geometry is invalid");
  }
  const started = performance.now();
  let sourceBytes = 0;
  for (let layerIndex = 0; layerIndex < GEMMA_VISION_LAYER_COUNT; layerIndex += 1) {
    const weights = await loadGemmaVisionLayerWeights(source, layerIndex);
    sourceBytes += weights.sourceBytes;
    const resources = await createGemmaVisionLayerResources(
      device,
      hidden,
      rows,
      positions,
      weights,
    );
    try {
      const encoder = device.createCommandEncoder({
        label: `Gemma vision encoder layer ${layerIndex}`,
      });
      encodeGemmaVisionLayer(encoder, resources);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
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
  };
}