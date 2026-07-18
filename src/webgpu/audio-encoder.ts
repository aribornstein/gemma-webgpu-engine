import {
  GEMMA_AUDIO_LAYER_COUNT,
  GemmaAudioWeightCache,
  loadGemmaAudioLayerWeights,
  type GemmaAudioTensorSource,
} from "../model/gemma-audio-weights";
import {
  createGemmaAudioLayerResources,
  destroyGemmaAudioLayerResources,
  encodeGemmaAudioLayer,
} from "./audio-layer";

const HIDDEN_SIZE = 1024;

export interface GemmaAudioEncoderProgress {
  layerIndex: number;
  completedLayers: number;
  layerSourceBytes: number;
}

export interface GemmaAudioEncoderResult {
  output: GPUBuffer;
  layers: number;
  sourceBytes: number;
  elapsedMilliseconds: number;
  weightLoadMilliseconds: number;
  resourceSetupMilliseconds: number;
  executionMilliseconds: number;
}

export async function runGemmaAudioEncoder(
  device: GPUDevice,
  source: GemmaAudioTensorSource,
  hidden: GPUBuffer,
  mask: Uint32Array,
  rows: number,
  onProgress?: (progress: GemmaAudioEncoderProgress) => void,
  signal?: AbortSignal,
  weightCache?: GemmaAudioWeightCache,
): Promise<GemmaAudioEncoderResult> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 || mask.length !== rows ||
      hidden.size < rows * HIDDEN_SIZE * 4) {
    throw new Error("Gemma audio encoder input geometry is invalid");
  }
  const started = performance.now();
  let sourceBytes = 0;
  let weightLoadMilliseconds = 0;
  let resourceSetupMilliseconds = 0;
  let executionMilliseconds = 0;
  for (let layerIndex = 0; layerIndex < GEMMA_AUDIO_LAYER_COUNT; layerIndex += 1) {
    throwIfAborted(signal);
    const weightLoadStarted = performance.now();
    const weights = await (weightCache
      ? weightCache.loadLayer(source, layerIndex)
      : loadGemmaAudioLayerWeights(source, layerIndex));
    weightLoadMilliseconds += performance.now() - weightLoadStarted;
    sourceBytes += weights.sourceBytes;
    throwIfAborted(signal);
    const setupStarted = performance.now();
    const resources = await createGemmaAudioLayerResources(
      device,
      hidden,
      mask,
      rows,
      weights,
    );
    resourceSetupMilliseconds += performance.now() - setupStarted;
    try {
      const executionStarted = performance.now();
      const encoder = device.createCommandEncoder({
        label: `Gemma audio encoder layer ${layerIndex}`,
      });
      encodeGemmaAudioLayer(encoder, resources);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      executionMilliseconds += performance.now() - executionStarted;
      throwIfAborted(signal);
    } finally {
      destroyGemmaAudioLayerResources(resources);
    }
    onProgress?.({
      layerIndex,
      completedLayers: layerIndex + 1,
      layerSourceBytes: weights.sourceBytes,
    });
  }
  return {
    output: hidden,
    layers: GEMMA_AUDIO_LAYER_COUNT,
    sourceBytes,
    elapsedMilliseconds: performance.now() - started,
    weightLoadMilliseconds,
    resourceSetupMilliseconds,
    executionMilliseconds,
  };
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Gemma audio encoding was cancelled", "AbortError");
}