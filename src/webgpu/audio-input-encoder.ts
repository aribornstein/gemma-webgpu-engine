import {
  GemmaAudioWeightCache,
  loadGemmaAudioGlobalWeights,
  type GemmaAudioTensorSource,
} from "../model/gemma-audio-weights";
import type { GemmaAudioFeatures } from "../runtime/gemma-audio-input";
import {
  runGemmaAudioEncoder,
  type GemmaAudioEncoderProgress,
} from "./audio-encoder";
import {
  createGemmaAudioPostprocessResources,
  destroyGemmaAudioPostprocessResources,
  encodeGemmaAudioPostprocess,
  type GemmaAudioPostprocessResources,
} from "./audio-postprocess";
import {
  createGemmaAudioSubsamplerResources,
  destroyGemmaAudioSubsamplerResources,
  encodeGemmaAudioSubsampler,
  getGemmaAudioSubsamplerPipelines,
  type GemmaAudioSubsamplerResources,
} from "./audio-subsampler";

export interface GemmaAudioEncodingResources {
  output: GPUBuffer;
  softTokenCount: number;
  paddedTokenCount: number;
  sourceBytes: number;
  elapsedMilliseconds: number;
  postprocess: GemmaAudioPostprocessResources;
}

export async function encodeGemmaAudioFeatures(
  device: GPUDevice,
  source: GemmaAudioTensorSource,
  input: GemmaAudioFeatures,
  onProgress?: (progress: GemmaAudioEncoderProgress) => void,
  signal?: AbortSignal,
  weightCache?: GemmaAudioWeightCache,
): Promise<GemmaAudioEncodingResources> {
  if (input.softTokenCount < 1) throw new Error("Gemma audio input has no valid soft tokens");
  const started = performance.now();
  throwIfAborted(signal);
  const globals = await (weightCache
    ? weightCache.loadGlobals(source)
    : loadGemmaAudioGlobalWeights(source));
  throwIfAborted(signal);
  const pipelines = await getGemmaAudioSubsamplerPipelines(device);
  let subsampler: GemmaAudioSubsamplerResources | null =
    createGemmaAudioSubsamplerResources(device, pipelines, input, globals.subsampler);
  let postprocess: GemmaAudioPostprocessResources | null = null;
  try {
    const subsampleCommand = device.createCommandEncoder({ label: "Gemma audio subsampler" });
    encodeGemmaAudioSubsampler(subsampleCommand, pipelines, subsampler);
    device.queue.submit([subsampleCommand.finish()]);
    await device.queue.onSubmittedWorkDone();
    throwIfAborted(signal);
    validatePrefixMask(subsampler.outputMask, input.softTokenCount);
    const encoderResult = await runGemmaAudioEncoder(
      device,
      source,
      subsampler.output,
      subsampler.outputMask,
      subsampler.outputRows,
      onProgress,
      signal,
      weightCache,
    );
    throwIfAborted(signal);
    postprocess = await createGemmaAudioPostprocessResources(
      device,
      encoderResult.output,
      subsampler.outputRows,
      globals,
    );
    const postprocessCommand = device.createCommandEncoder({
      label: "Gemma audio output projection",
    });
    encodeGemmaAudioPostprocess(postprocessCommand, postprocess);
    device.queue.submit([postprocessCommand.finish()]);
    await device.queue.onSubmittedWorkDone();
    throwIfAborted(signal);
    destroyGemmaAudioSubsamplerResources(subsampler);
    subsampler = null;
    return {
      output: postprocess.output,
      softTokenCount: input.softTokenCount,
      paddedTokenCount: postprocess.rows,
      sourceBytes: globals.sourceBytes + encoderResult.sourceBytes,
      elapsedMilliseconds: performance.now() - started,
      postprocess,
    };
  } catch (error) {
    if (postprocess) destroyGemmaAudioPostprocessResources(postprocess);
    throw error;
  } finally {
    if (subsampler) destroyGemmaAudioSubsamplerResources(subsampler);
  }
}

export function destroyGemmaAudioEncodingResources(
  resources: GemmaAudioEncodingResources,
): void {
  destroyGemmaAudioPostprocessResources(resources.postprocess);
}

function validatePrefixMask(mask: Uint32Array, expectedValid: number): void {
  const valid = mask.reduce((sum, value) => sum + Number(value !== 0), 0);
  if (valid !== expectedValid || mask.some((value, index) => value !== Number(index < valid))) {
    throw new Error("Gemma audio output mask is not a contiguous valid prefix");
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Gemma audio encoding was cancelled", "AbortError");
}