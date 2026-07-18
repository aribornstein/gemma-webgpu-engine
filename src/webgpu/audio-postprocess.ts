import type { GemmaAudioGlobalWeights } from "../model/gemma-audio-weights";
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

const AUDIO_HIDDEN_SIZE = 1024;
const TEXT_HIDDEN_SIZE = 1536;

export interface GemmaAudioPostprocessResources {
  output: GPUBuffer;
  rows: number;
  towerProjection: {
    pipeline: GemmaVisionF32DensePipeline;
    resources: GemmaVisionF32DenseResources;
  };
  biasPipeline: GPUComputePipeline;
  biasBindGroup: GPUBindGroup;
  normalized: {
    pipeline: GemmaPrefillRmsPipeline;
    resources: GemmaPrefillRmsResources;
  };
  embeddingProjection: {
    pipeline: GemmaVisionF32DensePipeline;
    resources: GemmaVisionF32DenseResources;
  };
  ownedBuffers: GPUBuffer[];
}

const biasPipelineCache = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export async function createGemmaAudioPostprocessResources(
  device: GPUDevice,
  hidden: GPUBuffer,
  rows: number,
  weights: GemmaAudioGlobalWeights,
): Promise<GemmaAudioPostprocessResources> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 ||
      hidden.size < rows * AUDIO_HIDDEN_SIZE * 4) {
    throw new Error("Gemma audio postprocess input geometry is invalid");
  }
  const ownedBuffers: GPUBuffer[] = [];
  const own = <T extends { ownedBuffers: GPUBuffer[] }>(resources: T): T => {
    ownedBuffers.push(...resources.ownedBuffers);
    return resources;
  };
  const allocate = (label: string): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: rows * TEXT_HIDDEN_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const upload = (label: string, values: Float32Array): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: values.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, values);
    ownedBuffers.push(buffer);
    return buffer;
  };
  const projected = allocate("Gemma audio tower projection");
  const normalizedOutput = allocate("Gemma audio normalized tower output");
  const output = allocate("Gemma audio language embeddings");
  try {
    const [towerPipeline, normPipeline, embeddingPipeline, biasPipeline] = await Promise.all([
      getGemmaVisionF32DensePipeline(device, rows, AUDIO_HIDDEN_SIZE, TEXT_HIDDEN_SIZE),
      getGemmaPrefillRmsPipeline(device, TEXT_HIDDEN_SIZE, false),
      getGemmaVisionF32DensePipeline(device, rows, TEXT_HIDDEN_SIZE, TEXT_HIDDEN_SIZE),
      getBiasPipeline(device),
    ]);
    const towerWeights = upload("Gemma audio tower output weights", weights.towerOutput.weight);
    const bias = upload("Gemma audio tower output bias", weights.towerOutput.bias);
    const embeddingWeights = upload(
      "Gemma audio embedding projection weights",
      weights.embeddingProjection,
    );
    const towerProjection = {
      pipeline: towerPipeline,
      resources: own(createGemmaVisionF32DenseResources(
        device,
        towerPipeline,
        hidden,
        towerWeights,
        projected,
      )),
    };
    const parameters = device.createBuffer({
      label: "Gemma audio output bias parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(parameters, 0, new Uint32Array([rows, 0, 0, 0]));
    ownedBuffers.push(parameters);
    const biasBindGroup = device.createBindGroup({
      layout: biasPipeline.getBindGroupLayout(0),
      entries: [binding(0, projected), binding(1, bias), binding(2, parameters)],
    });
    const normalized = {
      pipeline: normPipeline,
      resources: own(createGemmaPrefillRmsResources(
        device,
        normPipeline,
        rows,
        projected,
        null,
        normalizedOutput,
      )),
    };
    const embeddingProjection = {
      pipeline: embeddingPipeline,
      resources: own(createGemmaVisionF32DenseResources(
        device,
        embeddingPipeline,
        normalizedOutput,
        embeddingWeights,
        output,
      )),
    };
    return {
      output,
      rows,
      towerProjection,
      biasPipeline,
      biasBindGroup,
      normalized,
      embeddingProjection,
      ownedBuffers,
    };
  } catch (error) {
    for (const buffer of ownedBuffers.toReversed()) buffer.destroy();
    throw error;
  }
}

export function encodeGemmaAudioPostprocess(
  encoder: GPUCommandEncoder,
  resources: GemmaAudioPostprocessResources,
): void {
  encodeGemmaVisionF32Dense(
    encoder,
    resources.towerProjection.pipeline,
    resources.towerProjection.resources,
  );
  const biasPass = encoder.beginComputePass({ label: "Gemma audio output bias" });
  biasPass.setPipeline(resources.biasPipeline);
  biasPass.setBindGroup(0, resources.biasBindGroup);
  biasPass.dispatchWorkgroups(Math.ceil(resources.rows * TEXT_HIDDEN_SIZE / 256));
  biasPass.end();
  encodeGemmaPrefillRms(encoder, resources.normalized.pipeline, resources.normalized.resources);
  encodeGemmaVisionF32Dense(
    encoder,
    resources.embeddingProjection.pipeline,
    resources.embeddingProjection.resources,
  );
}

export function destroyGemmaAudioPostprocessResources(
  resources: GemmaAudioPostprocessResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

function getBiasPipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  const cached = biasPipelineCache.get(device);
  if (cached) return cached;
  const pending = device.createComputePipelineAsync({
    label: "Gemma audio output bias",
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: `struct Parameters { rows: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= parameters.rows * ${TEXT_HIDDEN_SIZE}u) { return; }
  values[index] += bias[index % ${TEXT_HIDDEN_SIZE}u];
}`,
      }),
      entryPoint: "main",
    },
  }).catch((error) => {
    biasPipelineCache.delete(device);
    throw error;
  });
  biasPipelineCache.set(device, pending);
  return pending;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}