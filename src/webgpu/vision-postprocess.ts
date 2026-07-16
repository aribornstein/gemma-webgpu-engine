import type { GemmaVisionProjectorWeights } from "../model/gemma-vision-weights";
import {
  createGemmaPrefillRmsResources,
  encodeGemmaPrefillRms,
  getGemmaPrefillRmsPipeline,
  type GemmaPrefillRmsPipeline,
  type GemmaPrefillRmsResources,
} from "./prefill-rms";
import {
  createGemmaVisionF32DenseResources,
  encodeGemmaVisionF32Dense,
  getGemmaVisionF32DensePipeline,
  type GemmaVisionF32DensePipeline,
  type GemmaVisionF32DenseResources,
} from "./vision-f32-dense";
import {
  createGemmaVisionPoolResources,
  encodeGemmaVisionPool,
  getGemmaVisionPoolPipeline,
  type GemmaVisionPoolResources,
} from "./vision-pool";

const VISION_HIDDEN_SIZE = 768;
const TEXT_HIDDEN_SIZE = 1536;

export interface GemmaVisionPostprocessResources {
  output: GPUBuffer;
  outputRows: number;
  poolPipeline: GPUComputePipeline;
  pool: GemmaVisionPoolResources;
  normPipeline: GemmaPrefillRmsPipeline;
  norm: GemmaPrefillRmsResources;
  projectorPipeline: GemmaVisionF32DensePipeline;
  projector: GemmaVisionF32DenseResources;
  ownedBuffers: GPUBuffer[];
}

export async function createGemmaVisionPostprocessResources(
  device: GPUDevice,
  encodedPatches: GPUBuffer,
  patchRows: number,
  patchColumns: number,
  weights: GemmaVisionProjectorWeights,
): Promise<GemmaVisionPostprocessResources> {
  const outputRows = patchRows * patchColumns / 9;
  if (!Number.isInteger(outputRows) || outputRows < 1 || outputRows > 280 ||
      weights.projection.length !== TEXT_HIDDEN_SIZE * VISION_HIDDEN_SIZE) {
    throw new Error("Gemma vision postprocess geometry is invalid");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const own = <T extends { ownedBuffers: GPUBuffer[] }>(resources: T): T => {
    ownedBuffers.push(...resources.ownedBuffers);
    return resources;
  };
  const allocate = (label: string, elements: number): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: elements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    ownedBuffers.push(buffer);
    return buffer;
  };
  try {
    const [poolPipeline, normPipeline, projectorPipeline] = await Promise.all([
      getGemmaVisionPoolPipeline(device),
      getGemmaPrefillRmsPipeline(device, VISION_HIDDEN_SIZE, false),
      getGemmaVisionF32DensePipeline(
        device,
        outputRows,
        VISION_HIDDEN_SIZE,
        TEXT_HIDDEN_SIZE,
      ),
    ]);
    const pooled = allocate("Gemma vision pooled features", outputRows * VISION_HIDDEN_SIZE);
    const normalized = allocate(
      "Gemma vision normalized pooled features",
      outputRows * VISION_HIDDEN_SIZE,
    );
    const output = allocate("Gemma vision language soft tokens", outputRows * TEXT_HIDDEN_SIZE);
    const projection = allocate(
      "Gemma vision language projector weights",
      weights.projection.length,
    );
    device.queue.writeBuffer(projection, 0, weights.projection);
    const pool = own(createGemmaVisionPoolResources(
      device,
      poolPipeline,
      encodedPatches,
      patchRows,
      patchColumns,
      pooled,
    ));
    const norm = own(createGemmaPrefillRmsResources(
      device,
      normPipeline,
      outputRows,
      pooled,
      null,
      normalized,
    ));
    const projector = own(createGemmaVisionF32DenseResources(
      device,
      projectorPipeline,
      normalized,
      projection,
      output,
    ));
    return {
      output,
      outputRows,
      poolPipeline,
      pool,
      normPipeline,
      norm,
      projectorPipeline,
      projector,
      ownedBuffers,
    };
  } catch (error) {
    for (const buffer of ownedBuffers.toReversed()) buffer.destroy();
    throw error;
  }
}

export function encodeGemmaVisionPostprocess(
  encoder: GPUCommandEncoder,
  resources: GemmaVisionPostprocessResources,
): void {
  encodeGemmaVisionPool(encoder, resources.poolPipeline, resources.pool);
  encodeGemmaPrefillRms(encoder, resources.normPipeline, resources.norm);
  encodeGemmaVisionF32Dense(encoder, resources.projectorPipeline, resources.projector);
}

export function destroyGemmaVisionPostprocessResources(
  resources: GemmaVisionPostprocessResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}