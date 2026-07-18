import type { CachedTensorPayload } from "./cached-safetensors";
import {
  float32LittleEndian,
  packedUint32,
} from "./gemma-layer-materializer";
import type { GemmaLayerTensorSource } from "./gemma-layer-weights";

export const GEMMA_AUDIO_LAYER_COUNT = 12;

export interface GemmaAudioProjectionWeights {
  bits: 2 | 4;
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  inputScale: number;
  outputScale: number;
  sourceBytes: number;
}

export interface GemmaAudioFeedForwardWeights {
  input: GemmaAudioProjectionWeights;
  output: GemmaAudioProjectionWeights;
  preNorm: Float32Array;
  postNorm: Float32Array;
}

export interface GemmaAudioLayerWeights {
  layerIndex: number;
  feedForward1: GemmaAudioFeedForwardWeights;
  feedForward2: GemmaAudioFeedForwardWeights;
  attention: {
    query: GemmaAudioProjectionWeights;
    key: GemmaAudioProjectionWeights;
    value: GemmaAudioProjectionWeights;
    output: GemmaAudioProjectionWeights;
    relativeKeyProjection: Float32Array;
    perDimensionScale: Float32Array;
  };
  convolution: {
    input: GemmaAudioProjectionWeights;
    output: GemmaAudioProjectionWeights;
    depthwise: Float32Array;
    preNorm: Float32Array;
    convolutionNorm: Float32Array;
  };
  norms: {
    preAttention: Float32Array;
    postAttention: Float32Array;
    output: Float32Array;
  };
  sourceBytes: number;
}

export interface GemmaAudioGlobalWeights {
  subsampler: {
    convolution0: Float32Array;
    norm0: Float32Array;
    convolution1: Float32Array;
    norm1: Float32Array;
    projection: Float32Array;
  };
  towerOutput: {
    weight: Float32Array;
    bias: Float32Array;
  };
  embeddingProjection: Float32Array;
  sourceBytes: number;
}

export interface GemmaAudioWeightCacheEstimate {
  loadedEntryCount: number;
  sourceBytes: number;
  materializedBytes: number;
}

export type GemmaAudioTensorSource = GemmaLayerTensorSource;

export class GemmaAudioWeightCache {
  private globals: Promise<GemmaAudioGlobalWeights> | null = null;
  private loadedGlobals: GemmaAudioGlobalWeights | null = null;
  private readonly layers = new Map<number, Promise<GemmaAudioLayerWeights>>();
  private readonly loadedLayers = new Map<number, GemmaAudioLayerWeights>();

  loadGlobals(source: GemmaAudioTensorSource): Promise<GemmaAudioGlobalWeights> {
    if (!this.globals) {
      const pending = loadGemmaAudioGlobalWeights(source).then((weights) => {
        if (this.globals === pending) this.loadedGlobals = weights;
        return weights;
      }).catch((error) => {
        if (this.globals === pending) this.globals = null;
        throw error;
      });
      this.globals = pending;
    }
    return this.globals;
  }

  loadLayer(
    source: GemmaAudioTensorSource,
    layerIndex: number,
  ): Promise<GemmaAudioLayerWeights> {
    let pending = this.layers.get(layerIndex);
    if (!pending) {
      pending = loadGemmaAudioLayerWeights(source, layerIndex).then((weights) => {
        if (this.layers.get(layerIndex) === pending) this.loadedLayers.set(layerIndex, weights);
        return weights;
      }).catch((error) => {
        if (this.layers.get(layerIndex) === pending) this.layers.delete(layerIndex);
        throw error;
      });
      this.layers.set(layerIndex, pending);
    }
    return pending;
  }

  estimateRetainedMemory(): GemmaAudioWeightCacheEstimate {
    const entries = [
      ...(this.loadedGlobals ? [this.loadedGlobals] : []),
      ...this.loadedLayers.values(),
    ];
    const buffers = new Set<ArrayBufferLike>();
    const seen = new Set<object>();
    const visit = (value: unknown): void => {
      if (typeof value !== "object" || value === null || seen.has(value)) return;
      seen.add(value);
      if (ArrayBuffer.isView(value)) {
        buffers.add(value.buffer);
        return;
      }
      for (const child of Object.values(value)) visit(child);
    };
    for (const entry of entries) visit(entry);
    return {
      loadedEntryCount: entries.length,
      sourceBytes: entries.reduce((total, entry) => total + entry.sourceBytes, 0),
      materializedBytes: Array.from(buffers, (buffer) => buffer.byteLength)
        .reduce((total, byteLength) => total + byteLength, 0),
    };
  }

  clear(): void {
    this.globals = null;
    this.loadedGlobals = null;
    this.layers.clear();
    this.loadedLayers.clear();
  }
}

export async function loadGemmaAudioLayerWeights(
  source: GemmaAudioTensorSource,
  layerIndex: number,
): Promise<GemmaAudioLayerWeights> {
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= GEMMA_AUDIO_LAYER_COUNT) {
    throw new Error("Gemma audio layer index is invalid");
  }
  const prefix = `model.audio_tower.layers.${layerIndex}`;
  const projectionSpecs = {
    feedForward1Input: [`${prefix}.feed_forward1.ffw_layer_1`, 2, 4096, 1024],
    feedForward1Output: [`${prefix}.feed_forward1.ffw_layer_2`, 2, 1024, 4096],
    feedForward2Input: [`${prefix}.feed_forward2.ffw_layer_1`, 2, 4096, 1024],
    feedForward2Output: [`${prefix}.feed_forward2.ffw_layer_2`, 2, 1024, 4096],
    query: [`${prefix}.self_attn.q_proj`, 2, 1024, 1024],
    key: [`${prefix}.self_attn.k_proj`, 2, 1024, 1024],
    value: [`${prefix}.self_attn.v_proj`, 2, 1024, 1024],
    attentionOutput: [`${prefix}.self_attn.post`, 2, 1024, 1024],
    convolutionInput: [`${prefix}.lconv1d.linear_start`, 4, 2048, 1024],
    convolutionOutput: [`${prefix}.lconv1d.linear_end`, 2, 1024, 1024],
  } as const;
  const normSpecs = {
    feedForward1Pre: `${prefix}.feed_forward1.pre_layer_norm.weight`,
    feedForward1Post: `${prefix}.feed_forward1.post_layer_norm.weight`,
    feedForward2Pre: `${prefix}.feed_forward2.pre_layer_norm.weight`,
    feedForward2Post: `${prefix}.feed_forward2.post_layer_norm.weight`,
    convolutionPre: `${prefix}.lconv1d.pre_layer_norm.weight`,
    convolution: `${prefix}.lconv1d.conv_norm.weight`,
    preAttention: `${prefix}.norm_pre_attn.weight`,
    postAttention: `${prefix}.norm_post_attn.weight`,
    output: `${prefix}.norm_out.weight`,
  } as const;
  const relativeKeyName = `${prefix}.self_attn.relative_k_proj.weight`;
  const perDimensionScaleName = `${prefix}.self_attn.per_dim_scale`;
  const depthwiseName = `${prefix}.lconv1d.depthwise_conv1d.weight`;

  const projectionNames = Object.values(projectionSpecs).flatMap(([name, bits, rows, columns]) =>
    validateProjection(source, name, bits, rows, columns));
  for (const name of Object.values(normSpecs)) validateDescriptor(source, name, "F32", [1024]);
  validateDescriptor(source, relativeKeyName, "F32", [1024, 1024]);
  validateDescriptor(source, perDimensionScaleName, "F32", [128]);
  validateDescriptor(source, depthwiseName, "F32", [1024, 1, 5]);
  const names = [
    ...projectionNames,
    ...Object.values(normSpecs),
    relativeKeyName,
    perDimensionScaleName,
    depthwiseName,
  ];
  if (names.length !== 52) throw new Error("Gemma audio layer contract is incomplete");
  const tensors = await source.readTensors(names);
  validateLoadedTensors(source, tensors, names, `Gemma audio layer ${layerIndex}`);
  const projection = (spec: readonly [string, 2 | 4, number, number]) =>
    materializeProjection(tensors, ...spec);
  const norm = (name: string) => float32LittleEndian(required(tensors, name));

  return {
    layerIndex,
    feedForward1: {
      input: projection(projectionSpecs.feedForward1Input),
      output: projection(projectionSpecs.feedForward1Output),
      preNorm: norm(normSpecs.feedForward1Pre),
      postNorm: norm(normSpecs.feedForward1Post),
    },
    feedForward2: {
      input: projection(projectionSpecs.feedForward2Input),
      output: projection(projectionSpecs.feedForward2Output),
      preNorm: norm(normSpecs.feedForward2Pre),
      postNorm: norm(normSpecs.feedForward2Post),
    },
    attention: {
      query: projection(projectionSpecs.query),
      key: projection(projectionSpecs.key),
      value: projection(projectionSpecs.value),
      output: projection(projectionSpecs.attentionOutput),
      relativeKeyProjection: norm(relativeKeyName),
      perDimensionScale: norm(perDimensionScaleName),
    },
    convolution: {
      input: projection(projectionSpecs.convolutionInput),
      output: projection(projectionSpecs.convolutionOutput),
      depthwise: norm(depthwiseName),
      preNorm: norm(normSpecs.convolutionPre),
      convolutionNorm: norm(normSpecs.convolution),
    },
    norms: {
      preAttention: norm(normSpecs.preAttention),
      postAttention: norm(normSpecs.postAttention),
      output: norm(normSpecs.output),
    },
    sourceBytes: names.reduce((sum, name) => sum + required(tensors, name).byteLength, 0),
  };
}

export async function loadGemmaAudioGlobalWeights(
  source: GemmaAudioTensorSource,
): Promise<GemmaAudioGlobalWeights> {
  const specs = {
    convolution0: [
      "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
      [128, 1, 3, 3],
    ],
    norm0: [
      "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
      [128],
    ],
    convolution1: [
      "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
      [32, 128, 3, 3],
    ],
    norm1: [
      "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
      [32],
    ],
    subsamplerProjection: [
      "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
      [1024, 1024],
    ],
    towerOutputWeight: ["model.audio_tower.output_proj.weight", [1536, 1024]],
    towerOutputBias: ["model.audio_tower.output_proj.bias", [1536]],
    embeddingProjection: ["model.embed_audio.embedding_projection.weight", [1536, 1536]],
  } as const;
  const names = Object.values(specs).map(([name, shape]) => {
    validateDescriptor(source, name, "F32", shape);
    return name;
  });
  const tensors = await source.readTensors(names);
  validateLoadedTensors(source, tensors, names, "Gemma audio globals");
  const read = (name: string) => float32LittleEndian(required(tensors, name));
  return {
    subsampler: {
      convolution0: read(specs.convolution0[0]),
      norm0: read(specs.norm0[0]),
      convolution1: read(specs.convolution1[0]),
      norm1: read(specs.norm1[0]),
      projection: read(specs.subsamplerProjection[0]),
    },
    towerOutput: {
      weight: read(specs.towerOutputWeight[0]),
      bias: read(specs.towerOutputBias[0]),
    },
    embeddingProjection: read(specs.embeddingProjection[0]),
    sourceBytes: names.reduce((sum, name) => sum + required(tensors, name).byteLength, 0),
  };
}

function validateProjection(
  source: GemmaAudioTensorSource,
  prefix: string,
  bits: 2 | 4,
  rows: number,
  columns: number,
): string[] {
  const names = [
    `${prefix}.linear.weight`,
    `${prefix}.linear.weight_scale`,
    `${prefix}.linear.input_activation_scale`,
    `${prefix}.linear.output_activation_scale`,
  ];
  validateDescriptor(source, names[0], "U8", [rows, columns * bits / 8]);
  validateDescriptor(source, names[1], "F32", [rows, 1]);
  validateDescriptor(source, names[2], "F32", []);
  validateDescriptor(source, names[3], "F32", []);
  return names;
}

function materializeProjection(
  tensors: ReadonlyMap<string, CachedTensorPayload>,
  prefix: string,
  bits: 2 | 4,
  _rows: number,
  _columns: number,
): GemmaAudioProjectionWeights {
  const weight = required(tensors, `${prefix}.linear.weight`);
  const rowScales = required(tensors, `${prefix}.linear.weight_scale`);
  const inputScale = float32LittleEndian(required(
    tensors,
    `${prefix}.linear.input_activation_scale`,
  ));
  const outputScale = float32LittleEndian(required(
    tensors,
    `${prefix}.linear.output_activation_scale`,
  ));
  if (inputScale.length !== 1 || outputScale.length !== 1) {
    throw new Error(`Gemma audio projection ${prefix} has invalid activation scales`);
  }
  return {
    bits,
    packedWeights: packedUint32(weight),
    rowScales: float32LittleEndian(rowScales),
    inputScale: inputScale[0],
    outputScale: outputScale[0],
    sourceBytes: weight.byteLength + rowScales.byteLength +
      inputScale.byteLength + outputScale.byteLength,
  };
}

function validateDescriptor(
  source: GemmaAudioTensorSource,
  name: string,
  dtype: string,
  shape: readonly number[],
): void {
  const descriptor = source.descriptors.get(name);
  if (!descriptor) throw new Error(`Gemma audio tensor ${name} is absent`);
  if (descriptor.dtype !== dtype || descriptor.shape.join(",") !== shape.join(",")) {
    throw new Error(`Gemma audio tensor ${name} does not match its pinned contract`);
  }
}

function validateLoadedTensors(
  source: GemmaAudioTensorSource,
  tensors: ReadonlyMap<string, CachedTensorPayload>,
  names: readonly string[],
  label: string,
): void {
  if (tensors.size !== names.length) {
    throw new Error(`${label} returned ${tensors.size} tensors; expected ${names.length}`);
  }
  for (const name of names) {
    const descriptor = source.descriptors.get(name);
    const tensor = tensors.get(name);
    if (!descriptor || !tensor || tensor.byteLength !== descriptor.byteLength ||
        tensor.bytes.byteLength !== descriptor.byteLength || tensor.sha256.length !== 64) {
      throw new Error(`${label} loaded payload mismatch for ${name}`);
    }
  }
}

function required(
  tensors: ReadonlyMap<string, CachedTensorPayload>,
  name: string,
): CachedTensorPayload {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Gemma audio tensor ${name} was not loaded`);
  return tensor;
}