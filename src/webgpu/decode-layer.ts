import type { GemmaLayerProfile } from "../model/gemma-layer-plan";
import type { DecodeOprojNormMode } from "./decode-oproj-norm";
import {
  compileDecodeAttentionBlockPipelines,
  encodeDecodeAttentionBlockPass,
  type DecodeAttentionBlockPipelines,
  type DecodeAttentionBlockResources,
} from "./decode-attention-block";
import {
  compileDecodeMlpPleBlockPipelines,
  encodeDecodeMlpPleBlockPass,
  type DecodeMlpPleBlockPipelines,
  type DecodeMlpPleBlockResources,
} from "./decode-mlp-ple-block";

export interface GemmaDecodeLayerPipelines {
  profile: GemmaLayerProfile;
  attention: DecodeAttentionBlockPipelines;
  mlp: DecodeMlpPleBlockPipelines;
}

export interface GemmaDecodeLayerResources {
  attention: DecodeAttentionBlockResources;
  mlp: DecodeMlpPleBlockResources;
}

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaDecodeLayerPipelines>>
>();

export function getGemmaDecodeLayerPipelines(
  device: GPUDevice,
  profile: GemmaLayerProfile,
  oprojMode: DecodeOprojNormMode = "subgroup-rows",
): Promise<GemmaDecodeLayerPipelines> {
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const cacheKey = `${profile}:${oprojMode}`;
  const cached = devicePipelines.get(cacheKey);
  if (cached) return cached;

  const compiled = compileGemmaDecodeLayerPipelines(device, profile, oprojMode).catch((error) => {
    devicePipelines?.delete(cacheKey);
    throw error;
  });
  devicePipelines.set(cacheKey, compiled);
  return compiled;
}

export async function compileGemmaDecodeLayerPipelines(
  device: GPUDevice,
  profile: GemmaLayerProfile,
  oprojMode: DecodeOprojNormMode = "subgroup-rows",
): Promise<GemmaDecodeLayerPipelines> {
  const [attention, mlp] = await Promise.all([
    compileDecodeAttentionBlockPipelines(device, profile, oprojMode),
    compileDecodeMlpPleBlockPipelines(device, profile),
  ]);
  return { profile, attention, mlp };
}

export function encodeGemmaDecodeLayer(
  encoder: GPUCommandEncoder,
  pipelines: GemmaDecodeLayerPipelines,
  resources: GemmaDecodeLayerResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma decode layer" });
  encodeGemmaDecodeLayerPass(pass, pipelines, resources);
  pass.end();
}

export function encodeGemmaDecodeLayerPass(
  pass: GPUComputePassEncoder,
  pipelines: GemmaDecodeLayerPipelines,
  resources: GemmaDecodeLayerResources,
): void {
  if (pipelines.profile !== pipelines.attention.profile ||
      pipelines.attention.headDim !== resources.attention.cache.headDim ||
      pipelines.mlp.bitWidth !== (pipelines.profile.endsWith("int2") ? 2 : 4)) {
    throw new Error("Gemma decode layer pipeline geometry is inconsistent");
  }
  encodeDecodeAttentionBlockPass(pass, pipelines.attention, resources.attention);
  encodeDecodeMlpPleBlockPass(pass, pipelines.mlp, resources.mlp);
}

export function gemmaDecodeLayerDispatchCount(
  resources: GemmaDecodeLayerResources,
): 7 | 8 | 9 | 10 {
  const attentionDispatches =
    Number(resources.attention.runsInputRms) +
    1 +
    (resources.attention.writesKvCache ? 2 : 0) +
    2;
  return (attentionDispatches + 4) as 7 | 8 | 9 | 10;
}