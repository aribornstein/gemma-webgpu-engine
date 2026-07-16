import { createGemmaGeluLut } from "../model/gemma-gelu-lut";
import { createGemmaVisionRotaryTable } from "../model/gemma-vision-rope";
import type {
  GemmaVisionLayerWeights,
  GemmaVisionProjectionWeights,
} from "../model/gemma-vision-weights";
import {
  createGemmaPrefillAttentionResources,
  encodeGemmaPrefillAttention,
  getGemmaPrefillAttentionPipeline,
  type GemmaPrefillAttentionPipeline,
  type GemmaPrefillAttentionResources,
} from "./prefill-attention";
import {
  createGemmaPrefillAddResources,
  createGemmaPrefillGeluMultiplyResources,
  encodeGemmaPrefillElementwise,
  getGemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwiseResources,
} from "./prefill-elementwise";
import {
  createGemmaPrefillPleDenseResources,
  encodeGemmaPrefillPleDense,
  getGemmaPrefillPleDensePipeline,
  type GemmaPrefillPleDensePipeline,
  type GemmaPrefillPleDenseResources,
} from "./prefill-ple-dense";
import {
  createGemmaPrefillRmsResources,
  encodeGemmaPrefillRms,
  getGemmaPrefillRmsPipeline,
  type GemmaPrefillRmsPipeline,
  type GemmaPrefillRmsResources,
} from "./prefill-rms";
import {
  createGemmaVisionRopeResources,
  encodeGemmaVisionRope,
  getGemmaVisionRopePipeline,
  updateGemmaVisionRope,
  type GemmaVisionRopeResources,
} from "./vision-rope";

const HIDDEN_SIZE = 768;
const INTERMEDIATE_SIZE = 3072;
const HEADS = 12;
const HEAD_DIMENSION = 64;

interface VisionNorm {
  pipeline: GemmaPrefillRmsPipeline;
  resources: GemmaPrefillRmsResources;
}

interface VisionDense {
  pipeline: GemmaPrefillPleDensePipeline;
  resources: GemmaPrefillPleDenseResources;
}

export interface GemmaVisionLayerResources {
  layerIndex: number;
  rows: number;
  output: GPUBuffer;
  inputNorm: VisionNorm;
  query: VisionDense;
  key: VisionDense;
  value: VisionDense;
  queryNorm: VisionNorm;
  keyNorm: VisionNorm;
  valueNorm: VisionNorm;
  queryRope: GemmaVisionRopeResources;
  keyRope: GemmaVisionRopeResources;
  ropePipeline: GPUComputePipeline;
  attention: GemmaPrefillAttentionResources;
  attentionPipeline: GemmaPrefillAttentionPipeline;
  attentionOutput: VisionDense;
  postAttentionNorm: VisionNorm;
  attentionResidual: GemmaPrefillElementwiseResources;
  preFeedforwardNorm: VisionNorm;
  gate: VisionDense;
  up: VisionDense;
  gateActivation: GemmaPrefillElementwiseResources;
  down: VisionDense;
  postFeedforwardNorm: VisionNorm;
  feedforwardResidual: GemmaPrefillElementwiseResources;
  elementwisePipelines: GemmaPrefillElementwisePipelines;
  ownedBuffers: GPUBuffer[];
}

export async function createGemmaVisionLayerResources(
  device: GPUDevice,
  hidden: GPUBuffer,
  rows: number,
  positions: Int32Array,
  weights: GemmaVisionLayerWeights,
): Promise<GemmaVisionLayerResources> {
  if (!Number.isInteger(rows) || rows < 1 || rows > 2520 ||
      positions.length < rows * 2 || hidden.size < rows * HIDDEN_SIZE * 4) {
    throw new Error("Gemma vision layer input geometry is invalid");
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
  const norm = async (
    dimension: number,
    rowCount: number,
    input: GPUBuffer,
    weight: Float32Array | null,
    output: GPUBuffer,
    label: string,
  ): Promise<VisionNorm> => {
    const pipeline = await getGemmaPrefillRmsPipeline(device, dimension, weight !== null);
    const weightBuffer = weight ? upload(`${label} weight`, weight) : null;
    return {
      pipeline,
      resources: own(createGemmaPrefillRmsResources(
        device,
        pipeline,
        rowCount,
        input,
        weightBuffer,
        output,
      )),
    };
  };
  const dense = async (
    input: GPUBuffer,
    projection: GemmaVisionProjectionWeights,
    inFeatures: number,
    outFeatures: number,
    output: GPUBuffer,
    label: string,
  ): Promise<VisionDense> => {
    const pipeline = await getGemmaPrefillPleDensePipeline(device, {
      rows,
      inFeatures,
      outFeatures,
    });
    const codes = upload(`${label} signed-I8 weights`, projection.packedWeights);
    const rowScales = upload(`${label} row scales`, projection.rowScales);
    return {
      pipeline,
      resources: own(createGemmaPrefillPleDenseResources(
        device,
        pipeline,
        input,
        {
          codes,
          rowScales,
          inputScale: projection.inputScale,
          outputScale: projection.outputScale,
        },
        output,
      )),
    };
  };

  const hiddenElements = rows * HIDDEN_SIZE;
  const headElements = rows * HEADS * HEAD_DIMENSION;
  const intermediateElements = rows * INTERMEDIATE_SIZE;
  const scratch = {
    inputNorm: allocate("Gemma vision input norm", hiddenElements),
    query: allocate("Gemma vision query", headElements),
    key: allocate("Gemma vision key", headElements),
    value: allocate("Gemma vision value", headElements),
    queryNorm: allocate("Gemma vision normalized query", headElements),
    keyNorm: allocate("Gemma vision normalized key", headElements),
    valueNorm: allocate("Gemma vision normalized value", headElements),
    attention: allocate("Gemma vision attention", hiddenElements),
    attentionOutput: allocate("Gemma vision attention projection", hiddenElements),
    postAttentionNorm: allocate("Gemma vision post-attention norm", hiddenElements),
    preFeedforwardNorm: allocate("Gemma vision pre-feedforward norm", hiddenElements),
    gate: allocate("Gemma vision MLP gate", intermediateElements),
    up: allocate("Gemma vision MLP up", intermediateElements),
    activated: allocate("Gemma vision activated gate", intermediateElements),
    down: allocate("Gemma vision MLP down", hiddenElements),
    postFeedforwardNorm: allocate("Gemma vision post-feedforward norm", hiddenElements),
  };

  try {
    const [elementwisePipelines, ropePipeline, attentionPipeline] = await Promise.all([
      getGemmaPrefillElementwisePipelines(device),
      getGemmaVisionRopePipeline(device),
      getGemmaPrefillAttentionPipeline(device, HEAD_DIMENSION),
    ]);
    const inputNorm = await norm(
      HIDDEN_SIZE,
      rows,
      hidden,
      weights.norms.input,
      scratch.inputNorm,
      "Gemma vision input norm",
    );
    const [query, key, value] = await Promise.all([
      dense(scratch.inputNorm, weights.query, HIDDEN_SIZE, HIDDEN_SIZE, scratch.query,
        "Gemma vision query"),
      dense(scratch.inputNorm, weights.key, HIDDEN_SIZE, HIDDEN_SIZE, scratch.key,
        "Gemma vision key"),
      dense(scratch.inputNorm, weights.value, HIDDEN_SIZE, HIDDEN_SIZE, scratch.value,
        "Gemma vision value"),
    ]);
    const [queryNorm, keyNorm, valueNorm] = await Promise.all([
      norm(HEAD_DIMENSION, rows * HEADS, scratch.query, weights.norms.query,
        scratch.queryNorm, "Gemma vision query norm"),
      norm(HEAD_DIMENSION, rows * HEADS, scratch.key, weights.norms.key,
        scratch.keyNorm, "Gemma vision key norm"),
      norm(HEAD_DIMENSION, rows * HEADS, scratch.value, null,
        scratch.valueNorm, "Gemma vision value norm"),
    ]);
    const rotary = createGemmaVisionRotaryTable(positions, rows);
    const queryRope = own(createGemmaVisionRopeResources(
      device,
      ropePipeline,
      scratch.queryNorm,
      rows,
      HEADS,
    ));
    const keyRope = own(createGemmaVisionRopeResources(
      device,
      ropePipeline,
      scratch.keyNorm,
      rows,
      HEADS,
    ));
    updateGemmaVisionRope(device, queryRope, rotary);
    updateGemmaVisionRope(device, keyRope, rotary);
    const attention = own(createGemmaPrefillAttentionResources(
      device,
      attentionPipeline,
      scratch.queryNorm,
      scratch.keyNorm,
      scratch.valueNorm,
      rows,
      rows,
      {
        sequence: rows,
        keyLength: rows,
        queryOffset: 0,
        queryHeads: HEADS,
        kvHeads: HEADS,
        window: 0,
        causal: false,
      },
      scratch.attention,
    ));
    const attentionOutput = await dense(
      scratch.attention,
      weights.attentionOutput,
      HIDDEN_SIZE,
      HIDDEN_SIZE,
      scratch.attentionOutput,
      "Gemma vision attention output",
    );
    const postAttentionNorm = await norm(
      HIDDEN_SIZE,
      rows,
      scratch.attentionOutput,
      weights.norms.postAttention,
      scratch.postAttentionNorm,
      "Gemma vision post-attention norm",
    );
    const attentionResidual = own(createGemmaPrefillAddResources(
      device,
      elementwisePipelines.add,
      hidden,
      scratch.postAttentionNorm,
      hiddenElements,
    ));
    const preFeedforwardNorm = await norm(
      HIDDEN_SIZE,
      rows,
      hidden,
      weights.norms.preFeedforward,
      scratch.preFeedforwardNorm,
      "Gemma vision pre-feedforward norm",
    );
    const [gate, up] = await Promise.all([
      dense(scratch.preFeedforwardNorm, weights.gate, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        scratch.gate, "Gemma vision MLP gate"),
      dense(scratch.preFeedforwardNorm, weights.up, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        scratch.up, "Gemma vision MLP up"),
    ]);
    const geluLookup = upload(
      "Gemma vision GELU lookup",
      createGemmaGeluLut(weights.gate.outputScale),
    );
    const gateActivation = own(createGemmaPrefillGeluMultiplyResources(
      device,
      elementwisePipelines.geluMultiply,
      scratch.gate,
      scratch.up,
      geluLookup,
      scratch.activated,
      intermediateElements,
      weights.gate.outputScale,
    ));
    const down = await dense(
      scratch.activated,
      weights.down,
      INTERMEDIATE_SIZE,
      HIDDEN_SIZE,
      scratch.down,
      "Gemma vision MLP down",
    );
    const postFeedforwardNorm = await norm(
      HIDDEN_SIZE,
      rows,
      scratch.down,
      weights.norms.postFeedforward,
      scratch.postFeedforwardNorm,
      "Gemma vision post-feedforward norm",
    );
    const feedforwardResidual = own(createGemmaPrefillAddResources(
      device,
      elementwisePipelines.add,
      hidden,
      scratch.postFeedforwardNorm,
      hiddenElements,
    ));
    return {
      layerIndex: weights.layerIndex,
      rows,
      output: hidden,
      inputNorm,
      query,
      key,
      value,
      queryNorm,
      keyNorm,
      valueNorm,
      queryRope,
      keyRope,
      ropePipeline,
      attention,
      attentionPipeline,
      attentionOutput,
      postAttentionNorm,
      attentionResidual,
      preFeedforwardNorm,
      gate,
      up,
      gateActivation,
      down,
      postFeedforwardNorm,
      feedforwardResidual,
      elementwisePipelines,
      ownedBuffers,
    };
  } catch (error) {
    for (const buffer of ownedBuffers.toReversed()) buffer.destroy();
    throw error;
  }
}

export function encodeGemmaVisionLayer(
  encoder: GPUCommandEncoder,
  resources: GemmaVisionLayerResources,
): void {
  encodeNorm(encoder, resources.inputNorm);
  encodeDense(encoder, resources.query);
  encodeDense(encoder, resources.key);
  encodeDense(encoder, resources.value);
  encodeNorm(encoder, resources.queryNorm);
  encodeNorm(encoder, resources.keyNorm);
  encodeNorm(encoder, resources.valueNorm);
  encodeGemmaVisionRope(
    encoder,
    resources.ropePipeline,
    resources.queryRope,
    resources.rows,
  );
  encodeGemmaVisionRope(
    encoder,
    resources.ropePipeline,
    resources.keyRope,
    resources.rows,
  );
  encodeGemmaPrefillAttention(
    encoder,
    resources.attentionPipeline,
    resources.attention,
    resources.rows,
  );
  encodeDense(encoder, resources.attentionOutput);
  encodeNorm(encoder, resources.postAttentionNorm);
  encodeGemmaPrefillElementwise(
    encoder,
    resources.elementwisePipelines.add,
    resources.attentionResidual,
  );
  encodeNorm(encoder, resources.preFeedforwardNorm);
  encodeDense(encoder, resources.gate);
  encodeDense(encoder, resources.up);
  encodeGemmaPrefillElementwise(
    encoder,
    resources.elementwisePipelines.geluMultiply,
    resources.gateActivation,
  );
  encodeDense(encoder, resources.down);
  encodeNorm(encoder, resources.postFeedforwardNorm);
  encodeGemmaPrefillElementwise(
    encoder,
    resources.elementwisePipelines.add,
    resources.feedforwardResidual,
  );
}

export function destroyGemmaVisionLayerResources(
  resources: GemmaVisionLayerResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

function encodeNorm(encoder: GPUCommandEncoder, norm: VisionNorm): void {
  encodeGemmaPrefillRms(encoder, norm.pipeline, norm.resources);
}

function encodeDense(encoder: GPUCommandEncoder, dense: VisionDense): void {
  encodeGemmaPrefillPleDense(encoder, dense.pipeline, dense.resources);
}