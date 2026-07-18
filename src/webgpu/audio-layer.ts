import type {
  GemmaAudioLayerWeights,
  GemmaAudioProjectionWeights,
} from "../model/gemma-audio-weights";
import {
  createGemmaAudioAttentionResources,
  createGemmaAudioRelativePositions,
  encodeGemmaAudioAttention,
  getGemmaAudioAttentionPipeline,
  type GemmaAudioAttentionResources,
} from "./audio-attention";
import {
  createGemmaAudioNormalizedResidualResources,
  createGemmaAudioSiluResources,
  encodeGemmaAudioNormalizedResidual,
  encodeGemmaAudioSilu,
  getGemmaAudioElementwisePipelines,
  type GemmaAudioElementwisePipelines,
  type GemmaAudioNormalizedResidualResources,
  type GemmaAudioSiluResources,
} from "./audio-elementwise";
import {
  createGemmaAudioLightConvolutionResources,
  encodeGemmaAudioLightConvolution,
  getGemmaAudioLightConvolutionPipeline,
  type GemmaAudioLightConvolutionResources,
} from "./audio-light-convolution";
import {
  createGemmaPrefillAddResources,
  encodeGemmaPrefillElementwise,
  getGemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwiseResources,
} from "./prefill-elementwise";
import {
  createGemmaPrefillQatLinearResources,
  encodeGemmaPrefillQatLinear,
  getGemmaPrefillQatLinearPipelines,
  type GemmaPrefillQatLinearPipelines,
  type GemmaPrefillQatLinearResources,
} from "./prefill-qat-linear";
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

const HIDDEN_SIZE = 1024;
const INTERMEDIATE_SIZE = 4096;

interface AudioNorm {
  pipeline: GemmaPrefillRmsPipeline;
  resources: GemmaPrefillRmsResources;
}

interface AudioQat {
  pipeline: GemmaPrefillQatLinearPipelines;
  resources: GemmaPrefillQatLinearResources;
}

interface AudioFeedForward {
  preNorm: AudioNorm;
  input: AudioQat;
  activation: GemmaAudioSiluResources;
  output: AudioQat;
  residual: GemmaAudioNormalizedResidualResources;
}

export interface GemmaAudioLayerResources {
  layerIndex: number;
  rows: number;
  output: GPUBuffer;
  feedForward1: AudioFeedForward;
  preAttentionNorm: AudioNorm;
  query: AudioQat;
  key: AudioQat;
  value: AudioQat;
  relativeProjection: {
    pipeline: GemmaVisionF32DensePipeline;
    resources: GemmaVisionF32DenseResources;
  };
  attentionPipeline: GPUComputePipeline;
  attention: GemmaAudioAttentionResources;
  attentionOutput: AudioQat;
  attentionResidual: GemmaAudioNormalizedResidualResources;
  convolutionPreNorm: AudioNorm;
  convolutionInput: AudioQat;
  lightConvolutionPipeline: GPUComputePipeline;
  lightConvolution: GemmaAudioLightConvolutionResources;
  convolutionOutput: AudioQat;
  convolutionResidual: GemmaPrefillElementwiseResources;
  feedForward2: AudioFeedForward;
  finalNorm: AudioNorm;
  audioElementwisePipelines: GemmaAudioElementwisePipelines;
  prefillElementwisePipelines: GemmaPrefillElementwisePipelines;
  ownedBuffers: GPUBuffer[];
}

export async function createGemmaAudioLayerResources(
  device: GPUDevice,
  hidden: GPUBuffer,
  mask: Uint32Array,
  rows: number,
  weights: GemmaAudioLayerWeights,
): Promise<GemmaAudioLayerResources> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 750 || mask.length !== rows ||
      hidden.size < rows * HIDDEN_SIZE * 4) {
    throw new Error("Gemma audio layer input geometry is invalid");
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
  const upload = (label: string, values: Float32Array | Uint32Array): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: values.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, values);
    ownedBuffers.push(buffer);
    return buffer;
  };
  const hiddenElements = rows * HIDDEN_SIZE;
  const scratch = {
    norm: allocate("Gemma audio normalized hidden", hiddenElements),
    wide: allocate("Gemma audio feed-forward intermediate", rows * INTERMEDIATE_SIZE),
    projected: allocate("Gemma audio projected hidden", hiddenElements),
    query: allocate("Gemma audio query", hiddenElements),
    key: allocate("Gemma audio key", hiddenElements),
    value: allocate("Gemma audio value", hiddenElements),
    relativeKeys: allocate("Gemma audio relative keys", 13 * HIDDEN_SIZE),
    attention: allocate("Gemma audio attention", hiddenElements),
    convolutionExpanded: allocate("Gemma audio convolution expanded", rows * 2048),
    convolutionActivated: allocate("Gemma audio convolution activated", hiddenElements),
    finalNorm: allocate("Gemma audio layer output norm", hiddenElements),
    srqHidden: allocate("Gemma audio hidden SRQ", hiddenElements),
    srqWide: allocate("Gemma audio wide SRQ", rows * INTERMEDIATE_SIZE),
    srqExpanded: allocate("Gemma audio convolution SRQ", rows * 2048),
  };

  try {
    const [
      audioElementwisePipelines,
      prefillElementwisePipelines,
      attentionPipeline,
      lightConvolutionPipeline,
      relativeProjectionPipeline,
    ] = await Promise.all([
      getGemmaAudioElementwisePipelines(device),
      getGemmaPrefillElementwisePipelines(device),
      getGemmaAudioAttentionPipeline(device),
      getGemmaAudioLightConvolutionPipeline(device),
      getGemmaVisionF32DensePipeline(device, 13, HIDDEN_SIZE, HIDDEN_SIZE),
    ]);
    const norm = async (
      input: GPUBuffer,
      normWeights: Float32Array,
      output: GPUBuffer,
      label: string,
    ): Promise<AudioNorm> => {
      const pipeline = await getGemmaPrefillRmsPipeline(device, HIDDEN_SIZE, true);
      const weightBuffer = upload(`${label} weights`, normWeights);
      return {
        pipeline,
        resources: own(createGemmaPrefillRmsResources(
          device,
          pipeline,
          rows,
          input,
          weightBuffer,
          output,
        )),
      };
    };
    const qat = async (
      input: GPUBuffer,
      projection: GemmaAudioProjectionWeights,
      inFeatures: number,
      outFeatures: number,
      output: GPUBuffer,
      srq: GPUBuffer,
      label: string,
    ): Promise<AudioQat> => {
      const pipeline = await getGemmaPrefillQatLinearPipelines(device, {
        rows,
        inFeatures,
        outFeatures,
        bits: projection.bits,
      });
      const packedWeights = upload(`${label} packed weights`, projection.packedWeights);
      const rowScales = upload(`${label} row scales`, projection.rowScales);
      return {
        pipeline,
        resources: own(createGemmaPrefillQatLinearResources(
          device,
          pipeline,
          input,
          {
            packedWeights,
            rowScales,
            inputScale: projection.inputScale,
            outputScale: projection.outputScale,
          },
          output,
          srq,
        )),
      };
    };
    const feedForward = async (
      feedForwardWeights: GemmaAudioLayerWeights["feedForward1"],
      label: string,
    ): Promise<AudioFeedForward> => {
      const preNorm = await norm(hidden, feedForwardWeights.preNorm, scratch.norm, `${label} pre`);
      const input = await qat(
        scratch.norm,
        feedForwardWeights.input,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        scratch.wide,
        scratch.srqHidden,
        `${label} input`,
      );
      const activation = own(createGemmaAudioSiluResources(
        device,
        audioElementwisePipelines.silu,
        scratch.wide,
        rows * INTERMEDIATE_SIZE,
      ));
      const output = await qat(
        scratch.wide,
        feedForwardWeights.output,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        scratch.projected,
        scratch.srqWide,
        `${label} output`,
      );
      const postNormWeights = upload(`${label} post norm weights`, feedForwardWeights.postNorm);
      const residual = own(createGemmaAudioNormalizedResidualResources(
        device,
        audioElementwisePipelines.normalizedResidual,
        scratch.projected,
        postNormWeights,
        hidden,
        rows,
        0.5,
      ));
      return { preNorm, input, activation, output, residual };
    };

    const feedForward1 = await feedForward(weights.feedForward1, "Gemma audio FFN 1");
    const preAttentionNorm = await norm(
      hidden,
      weights.norms.preAttention,
      scratch.norm,
      "Gemma audio pre-attention",
    );
    const [query, key, value] = await Promise.all([
      qat(scratch.norm, weights.attention.query, HIDDEN_SIZE, HIDDEN_SIZE,
        scratch.query, scratch.srqHidden, "Gemma audio query"),
      qat(scratch.norm, weights.attention.key, HIDDEN_SIZE, HIDDEN_SIZE,
        scratch.key, scratch.srqHidden, "Gemma audio key"),
      qat(scratch.norm, weights.attention.value, HIDDEN_SIZE, HIDDEN_SIZE,
        scratch.value, scratch.srqHidden, "Gemma audio value"),
    ]);
    const relativePositions = upload(
      "Gemma audio relative positions",
      createGemmaAudioRelativePositions(),
    );
    const relativeProjectionWeights = upload(
      "Gemma audio relative projection weights",
      weights.attention.relativeKeyProjection,
    );
    const relativeProjection = {
      pipeline: relativeProjectionPipeline,
      resources: own(createGemmaVisionF32DenseResources(
        device,
        relativeProjectionPipeline,
        relativePositions,
        relativeProjectionWeights,
        scratch.relativeKeys,
      )),
    };
    const perDimensionScale = upload(
      "Gemma audio per-dimension scale",
      weights.attention.perDimensionScale,
    );
    const maskBuffer = upload("Gemma audio attention mask", mask);
    const attention = own(createGemmaAudioAttentionResources(
      device,
      attentionPipeline,
      scratch.query,
      scratch.key,
      scratch.value,
      scratch.relativeKeys,
      perDimensionScale,
      maskBuffer,
      rows,
      scratch.attention,
    ));
    const attentionOutput = await qat(
      scratch.attention,
      weights.attention.output,
      HIDDEN_SIZE,
      HIDDEN_SIZE,
      scratch.projected,
      scratch.srqHidden,
      "Gemma audio attention output",
    );
    const postAttentionWeights = upload(
      "Gemma audio post-attention norm weights",
      weights.norms.postAttention,
    );
    const attentionResidual = own(createGemmaAudioNormalizedResidualResources(
      device,
      audioElementwisePipelines.normalizedResidual,
      scratch.projected,
      postAttentionWeights,
      hidden,
      rows,
      1,
    ));
    const convolutionPreNorm = await norm(
      hidden,
      weights.convolution.preNorm,
      scratch.norm,
      "Gemma audio convolution pre",
    );
    const convolutionInput = await qat(
      scratch.norm,
      weights.convolution.input,
      HIDDEN_SIZE,
      2048,
      scratch.convolutionExpanded,
      scratch.srqHidden,
      "Gemma audio convolution input",
    );
    const depthwiseWeights = upload(
      "Gemma audio depthwise convolution weights",
      weights.convolution.depthwise,
    );
    const convolutionNormWeights = upload(
      "Gemma audio convolution norm weights",
      weights.convolution.convolutionNorm,
    );
    const lightConvolution = own(createGemmaAudioLightConvolutionResources(
      device,
      lightConvolutionPipeline,
      scratch.convolutionExpanded,
      depthwiseWeights,
      convolutionNormWeights,
      rows,
      scratch.convolutionActivated,
    ));
    const convolutionOutput = await qat(
      scratch.convolutionActivated,
      weights.convolution.output,
      HIDDEN_SIZE,
      HIDDEN_SIZE,
      scratch.projected,
      scratch.srqHidden,
      "Gemma audio convolution output",
    );
    const convolutionResidual = own(createGemmaPrefillAddResources(
      device,
      prefillElementwisePipelines.add,
      hidden,
      scratch.projected,
      hiddenElements,
    ));
    const feedForward2 = await feedForward(weights.feedForward2, "Gemma audio FFN 2");
    const finalNorm = await norm(
      hidden,
      weights.norms.output,
      scratch.finalNorm,
      "Gemma audio output",
    );
    return {
      layerIndex: weights.layerIndex,
      rows,
      output: hidden,
      feedForward1,
      preAttentionNorm,
      query,
      key,
      value,
      relativeProjection,
      attentionPipeline,
      attention,
      attentionOutput,
      attentionResidual,
      convolutionPreNorm,
      convolutionInput,
      lightConvolutionPipeline,
      lightConvolution,
      convolutionOutput,
      convolutionResidual,
      feedForward2,
      finalNorm,
      audioElementwisePipelines,
      prefillElementwisePipelines,
      ownedBuffers,
    };
  } catch (error) {
    for (const buffer of ownedBuffers.toReversed()) buffer.destroy();
    throw error;
  }
}

export function encodeGemmaAudioLayer(
  encoder: GPUCommandEncoder,
  resources: GemmaAudioLayerResources,
): void {
  encodeFeedForward(encoder, resources, resources.feedForward1);
  encodeNorm(encoder, resources.preAttentionNorm);
  encodeQat(encoder, resources.query);
  encodeQat(encoder, resources.key);
  encodeQat(encoder, resources.value);
  encodeGemmaVisionF32Dense(
    encoder,
    resources.relativeProjection.pipeline,
    resources.relativeProjection.resources,
  );
  encodeGemmaAudioAttention(encoder, resources.attentionPipeline, resources.attention);
  encodeQat(encoder, resources.attentionOutput);
  encodeGemmaAudioNormalizedResidual(
    encoder,
    resources.audioElementwisePipelines,
    resources.attentionResidual,
  );
  encodeNorm(encoder, resources.convolutionPreNorm);
  encodeQat(encoder, resources.convolutionInput);
  encodeGemmaAudioLightConvolution(
    encoder,
    resources.lightConvolutionPipeline,
    resources.lightConvolution,
  );
  encodeQat(encoder, resources.convolutionOutput);
  encodeGemmaPrefillElementwise(
    encoder,
    resources.prefillElementwisePipelines.add,
    resources.convolutionResidual,
  );
  encodeFeedForward(encoder, resources, resources.feedForward2);
  encodeNorm(encoder, resources.finalNorm);
  encoder.copyBufferToBuffer(
    resources.finalNorm.resources.output,
    0,
    resources.output,
    0,
    resources.rows * HIDDEN_SIZE * 4,
  );
}

export function destroyGemmaAudioLayerResources(resources: GemmaAudioLayerResources): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

function encodeFeedForward(
  encoder: GPUCommandEncoder,
  layer: GemmaAudioLayerResources,
  feedForward: AudioFeedForward,
): void {
  encodeNorm(encoder, feedForward.preNorm);
  encodeQat(encoder, feedForward.input);
  encodeGemmaAudioSilu(encoder, layer.audioElementwisePipelines, feedForward.activation);
  encodeQat(encoder, feedForward.output);
  encodeGemmaAudioNormalizedResidual(
    encoder,
    layer.audioElementwisePipelines,
    feedForward.residual,
  );
}

function encodeNorm(encoder: GPUCommandEncoder, norm: AudioNorm): void {
  encodeGemmaPrefillRms(encoder, norm.pipeline, norm.resources);
}

function encodeQat(encoder: GPUCommandEncoder, projection: AudioQat): void {
  encodeGemmaPrefillQatLinear(encoder, projection.pipeline, projection.resources);
}