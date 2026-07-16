import {
  loadGemmaVisionPatchWeights,
  loadGemmaVisionProjectorWeights,
} from "../model/gemma-vision-weights";
import { PinnedSafetensorsSource } from "../model/pinned-safetensors";
import type { GemmaVisionInput } from "../runtime/gemma-vision-input";
import { runGemmaVisionEncoder, type GemmaVisionEncoderProgress } from "./vision-encoder";
import {
  createGemmaVisionPatchEmbedResources,
  encodeGemmaVisionPatchEmbed,
  getGemmaVisionPatchEmbedPipeline,
  updateGemmaVisionPatchEmbed,
} from "./vision-patch-embed";
import {
  createGemmaVisionPostprocessResources,
  destroyGemmaVisionPostprocessResources,
  encodeGemmaVisionPostprocess,
  type GemmaVisionPostprocessResources,
} from "./vision-postprocess";

const PATCH_DIMENSION = 768;
const VISION_HIDDEN_SIZE = 768;

export interface GemmaVisionImageResources {
  output: GPUBuffer;
  softTokenCount: number;
  patchCount: number;
  sourceBytes: number;
  elapsedMilliseconds: number;
  postprocess: GemmaVisionPostprocessResources;
}

export async function encodeGemmaVisionImage(
  device: GPUDevice,
  source: PinnedSafetensorsSource,
  input: GemmaVisionInput,
  onProgress?: (progress: GemmaVisionEncoderProgress) => void,
): Promise<GemmaVisionImageResources> {
  if (input.patchCount !== input.patchRows * input.patchColumns ||
      input.softTokenCount !== input.patchCount / 9 ||
      input.patches.length < input.patchCount * PATCH_DIMENSION ||
      input.positions.length < input.patchCount * 2) {
    throw new Error("Gemma vision image input geometry is invalid");
  }
  const started = performance.now();
  const patchWeights = await loadGemmaVisionPatchWeights(source);
  const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, size: number, usage = storageUpload) =>
    device.createBuffer({ label, size, usage });
  const patches = make("Gemma vision image patches", input.patchCount * PATCH_DIMENSION * 4);
  const positions = make("Gemma vision image positions", input.patchCount * 2 * 4);
  const projection = make(
    "Gemma vision patch projection",
    patchWeights.projection.byteLength,
  );
  const positionEmbeddings = make(
    "Gemma vision position embeddings",
    patchWeights.positions.byteLength,
  );
  const hidden = make(
    "Gemma vision encoded patches",
    input.patchCount * VISION_HIDDEN_SIZE * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const uploadBuffers = [patches, positions, projection, positionEmbeddings];
  let postprocess: GemmaVisionPostprocessResources | null = null;
  try {
    device.queue.writeBuffer(
      patches,
      0,
      input.patches.subarray(0, input.patchCount * PATCH_DIMENSION),
    );
    device.queue.writeBuffer(
      positions,
      0,
      input.positions.subarray(0, input.patchCount * 2),
    );
    device.queue.writeBuffer(projection, 0, patchWeights.projection);
    device.queue.writeBuffer(positionEmbeddings, 0, patchWeights.positions);
    const patchPipeline = await getGemmaVisionPatchEmbedPipeline(device);
    const patchResources = createGemmaVisionPatchEmbedResources(
      device,
      patchPipeline,
      patches,
      positions,
      projection,
      positionEmbeddings,
      input.patchCount,
      hidden,
    );
    try {
      updateGemmaVisionPatchEmbed(device, patchResources, input.patchCount);
      const command = device.createCommandEncoder({ label: "Gemma vision patch embedding" });
      encodeGemmaVisionPatchEmbed(command, patchPipeline, patchResources, input.patchCount);
      device.queue.submit([command.finish()]);
      await device.queue.onSubmittedWorkDone();
    } finally {
      for (const buffer of patchResources.ownedBuffers.toReversed()) buffer.destroy();
    }
    for (const buffer of uploadBuffers) buffer.destroy();
    uploadBuffers.length = 0;
    const encoderResult = await runGemmaVisionEncoder(
      device,
      source,
      hidden,
      input.patchCount,
      input.positions,
      onProgress,
    );
    const projectorWeights = await loadGemmaVisionProjectorWeights(source);
    postprocess = await createGemmaVisionPostprocessResources(
      device,
      hidden,
      input.patchRows,
      input.patchColumns,
      projectorWeights,
    );
    const command = device.createCommandEncoder({ label: "Gemma vision soft-token projection" });
    encodeGemmaVisionPostprocess(command, postprocess);
    device.queue.submit([command.finish()]);
    await device.queue.onSubmittedWorkDone();
    hidden.destroy();
    return {
      output: postprocess.output,
      softTokenCount: postprocess.outputRows,
      patchCount: input.patchCount,
      sourceBytes: patchWeights.sourceBytes + encoderResult.sourceBytes +
        projectorWeights.sourceBytes,
      elapsedMilliseconds: performance.now() - started,
      postprocess,
    };
  } catch (error) {
    if (postprocess) destroyGemmaVisionPostprocessResources(postprocess);
    hidden.destroy();
    throw error;
  } finally {
    for (const buffer of uploadBuffers) buffer.destroy();
  }
}

export function destroyGemmaVisionImageResources(
  resources: GemmaVisionImageResources,
): void {
  destroyGemmaVisionPostprocessResources(resources.postprocess);
}