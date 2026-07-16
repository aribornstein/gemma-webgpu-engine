import type { GemmaRotaryBlock } from "../model/gemma-rope";
import {
  createGemmaDecodeInputResources,
  encodeGemmaDecodeInputPass,
  type GemmaDecodeInputResources,
} from "./decode-input";
import {
  encodeGemmaGreedyPass,
  readGemmaGreedyResult,
} from "./decode-greedy";
import type {
  GemmaDecodeModelResources,
  GemmaModelOutput,
  GemmaModelOutputMode,
} from "./decode-model";
import {
  createGemmaPrefillAttentionResources,
  encodeGemmaPrefillAttentionPass,
  getGemmaPrefillAttentionPipeline,
  updateGemmaPrefillAttention,
  type GemmaPrefillAttentionPipeline,
  type GemmaPrefillAttentionResources,
} from "./prefill-attention";
import {
  createGemmaPrefillAddResources,
  createGemmaPrefillGeluMultiplyResources,
  createGemmaPrefillMultiplyResources,
  createGemmaPrefillStridedGeluMultiplyResources,
  encodeGemmaPrefillElementwisePass,
  getGemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwisePipelines,
  type GemmaPrefillElementwiseResources,
} from "./prefill-elementwise";
import {
  createGemmaPrefillPleDenseResources,
  encodeGemmaPrefillPleDensePass,
  getGemmaPrefillPleDensePipeline,
  type GemmaPrefillPleDensePipeline,
  type GemmaPrefillPleDenseResources,
} from "./prefill-ple-dense";
import { GemmaPrefillParameterArena } from "./prefill-parameter-arena";
import {
  createGemmaPrefillQatGateUpActivationResources,
  createGemmaPrefillQatGateUpResources,
  encodeGemmaPrefillQatGateUpActivationPass,
  encodeGemmaPrefillQatGateUpPass,
  getGemmaPrefillQatGateUpActivationPipelines,
  getGemmaPrefillQatGateUpPipelines,
  type GemmaPrefillQatGateUpActivationResources,
  type GemmaPrefillQatGateUpPipelines,
  type GemmaPrefillQatGateUpResources,
} from "./prefill-qat-gate-up";
import {
  createGemmaPrefillQatLinearResources,
  encodeGemmaPrefillQatLinearPass,
  getGemmaPrefillQatLinearPipelines,
  type GemmaPrefillBufferSlice,
  type GemmaPrefillQatLinearPipelines,
  type GemmaPrefillQatLinearResources,
  type GemmaPrefillQatLinearWeights,
} from "./prefill-qat-linear";
import {
  createGemmaPrefillRmsResidualResources,
  createGemmaPrefillRmsResources,
  encodeGemmaPrefillRmsResidualPass,
  encodeGemmaPrefillRmsPass,
  getGemmaPrefillRmsPipeline,
  getGemmaPrefillRmsResidualPipeline,
  type GemmaPrefillRmsBufferSlice,
  type GemmaPrefillRmsPipeline,
  type GemmaPrefillRmsResidualResources,
  type GemmaPrefillRmsResources,
} from "./prefill-rms";
import {
  createGemmaPrefillRopeResources,
  encodeGemmaPrefillRopePass,
  getGemmaPrefillRopePipeline,
  updateGemmaPrefillRope,
  type GemmaPrefillRopePipeline,
  type GemmaPrefillRopeResources,
} from "./prefill-rope";
import {
  createGemmaPrefillStridedCopyResources,
  encodeGemmaPrefillStridedCopyPass,
  getGemmaPrefillStridedCopyPipeline,
  updateGemmaPrefillStridedCopy,
  type GemmaPrefillStridedCopyResources,
} from "./prefill-strided-copy";

export const GEMMA_FIXED_PREFILL_ROWS = 32;
export type GemmaPrefillGateUpMode = "fused-activated" | "fused" | "separate";
export type GemmaPrefillRmsEpilogueMode = "fused" | "separate";
export type GemmaPrefillQkvSrqMode = "shared" | "separate";
export type GemmaPrefillPleInputMode = "direct" | "copied";

const LAYER_COUNT = 35;
const HIDDEN_SIZE = 1536;
const PLE_SIZE = 256;
const QUERY_HEADS = 8;
const HIDDEN_BYTES = HIDDEN_SIZE * 4;
const MAX_HEAD_OUTPUT = 4096;
const MAX_INTERMEDIATE = 12288;
const VOCAB_SIZE = 262144;

export type GemmaFixedPrefillGpuStage =
  | "input"
  | "attention"
  | "feedforward"
  | "ple"
  | "output";

export interface GemmaFixedPrefillGpuSample {
  stage: GemmaFixedPrefillGpuStage;
  layerIndex: number | null;
  gpuMs: number;
}

export interface GemmaFixedPrefillGpuProfile {
  position: number;
  validRows: number;
  samples: readonly GemmaFixedPrefillGpuSample[];
  stageGpuMs: Readonly<Record<GemmaFixedPrefillGpuStage, number>>;
  totalGpuMs: number;
}

export interface GemmaFixedPrefillSubmission extends GemmaModelOutput {
  gpuProfile: GemmaFixedPrefillGpuProfile | null;
}

interface GemmaFixedPrefillProfileRecord {
  stage: GemmaFixedPrefillGpuStage;
  layerIndex: number | null;
}

interface GemmaFixedPrefillProfileResources {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  records: GemmaFixedPrefillProfileRecord[];
}

interface GemmaPrefillProjection {
  pipelines: GemmaPrefillQatLinearPipelines;
  resources: GemmaPrefillQatLinearResources;
}

interface GemmaPrefillGateUpProjection {
  pipelines: GemmaPrefillQatGateUpPipelines;
  resources: GemmaPrefillQatGateUpResources;
}

interface GemmaPrefillGateUpActivationProjection {
  pipelines: GemmaPrefillQatGateUpPipelines;
  resources: GemmaPrefillQatGateUpActivationResources;
}

interface GemmaPrefillNorm {
  pipeline: GemmaPrefillRmsPipeline;
  resources: GemmaPrefillRmsResources;
}

interface GemmaPrefillNormResidual {
  pipeline: GemmaPrefillRmsPipeline;
  resources: GemmaPrefillRmsResidualResources;
}

interface GemmaPrefillRope {
  pipeline: GemmaPrefillRopePipeline;
  resources: GemmaPrefillRopeResources;
}

interface GemmaPrefillPleProjection {
  pipeline: GemmaPrefillPleDensePipeline;
  resources: GemmaPrefillPleDenseResources;
}

interface GemmaPrefillLayerResources {
  layerIndex: number;
  headDimension: 256 | 512;
  kvFeatures: 256 | 512;
  window: number;
  cacheCapacity: number;
  inputNorm: GemmaPrefillNorm;
  query: GemmaPrefillProjection;
  queryNorm: GemmaPrefillNorm;
  queryRope: GemmaPrefillRope;
  key: GemmaPrefillProjection | null;
  keyNorm: GemmaPrefillNorm | null;
  keyRope: GemmaPrefillRope | null;
  value: GemmaPrefillProjection | null;
  valueNorm: GemmaPrefillNorm | null;
  keyCopy: GemmaPrefillStridedCopyResources | null;
  valueCopy: GemmaPrefillStridedCopyResources | null;
  attentionPipeline: GemmaPrefillAttentionPipeline;
  attention: GemmaPrefillAttentionResources;
  outputProjection: GemmaPrefillProjection;
  postAttentionNorm: GemmaPrefillNorm | null;
  attentionResidual: GemmaPrefillElementwiseResources | null;
  attentionNormResidual: GemmaPrefillNormResidual | null;
  preFeedforwardNorm: GemmaPrefillNorm;
  gateUpActivation: GemmaPrefillGateUpActivationProjection | null;
  gateUp: GemmaPrefillGateUpProjection | null;
  gate: GemmaPrefillProjection | null;
  up: GemmaPrefillProjection | null;
  gateActivation: GemmaPrefillElementwiseResources | null;
  down: GemmaPrefillProjection;
  postFeedforwardNorm: GemmaPrefillNorm | null;
  feedforwardResidual: GemmaPrefillElementwiseResources | null;
  feedforwardNormResidual: GemmaPrefillNormResidual | null;
  pleInputCopy: GemmaPrefillStridedCopyResources | null;
  pleGate: GemmaPrefillPleProjection;
  pleActivation: GemmaPrefillElementwiseResources;
  pleProjection: GemmaPrefillPleProjection;
  postPleNorm: GemmaPrefillNorm | null;
  pleResidual: GemmaPrefillElementwiseResources | null;
  layerScale: GemmaPrefillElementwiseResources | null;
  pleNormResidualScale: GemmaPrefillNormResidual | null;
}

export interface GemmaFixedPrefillResources {
  input: GemmaDecodeInputResources;
  layers: readonly GemmaPrefillLayerResources[];
  finalNorm: GemmaPrefillNorm;
  lastRow: GPUBuffer;
  lastRowCopy: GemmaPrefillStridedCopyResources;
  lmHead: GemmaPrefillProjection;
  stridedCopyPipeline: GPUComputePipeline;
  elementwisePipelines: GemmaPrefillElementwisePipelines;
  decode: GemmaDecodeModelResources;
  cacheCapacity: number;
  ownedBuffers: GPUBuffer[];
}

interface GemmaPrefillScratch {
  hiddenNorm: GPUBuffer;
  residualNorm: GPUBuffer;
  query: GPUBuffer;
  queryNorm: GPUBuffer;
  key: GPUBuffer;
  keyNorm: GPUBuffer;
  value: GPUBuffer;
  valueNorm: GPUBuffer;
  attention: GPUBuffer;
  projection: GPUBuffer;
  gate: GPUBuffer;
  up: GPUBuffer;
  activated: GPUBuffer;
  down: GPUBuffer;
  pleInput: GPUBuffer;
  pleGate: GPUBuffer;
  pleActivated: GPUBuffer;
  pleProjection: GPUBuffer;
  srq: GPUBuffer;
  finalNorm: GPUBuffer;
}

export async function createGemmaFixedPrefillResources(
  device: GPUDevice,
  decode: GemmaDecodeModelResources,
  gateUpMode: GemmaPrefillGateUpMode = "fused",
  rmsEpilogueMode: GemmaPrefillRmsEpilogueMode = "fused",
  qkvSrqMode: GemmaPrefillQkvSrqMode = "separate",
  pleInputMode: GemmaPrefillPleInputMode = "copied",
): Promise<GemmaFixedPrefillResources> {
  if (gateUpMode !== "fused-activated" && gateUpMode !== "fused" &&
      gateUpMode !== "separate") {
    throw new Error("Gemma prefill gate/up mode is invalid");
  }
  if (rmsEpilogueMode !== "fused" && rmsEpilogueMode !== "separate") {
    throw new Error("Gemma prefill RMS epilogue mode is invalid");
  }
  if (qkvSrqMode !== "shared" && qkvSrqMode !== "separate") {
    throw new Error("Gemma prefill QKV SRQ mode is invalid");
  }
  if (pleInputMode !== "direct" && pleInputMode !== "copied") {
    throw new Error("Gemma prefill PLE input mode is invalid");
  }
  if (decode.stack.layers.length !== LAYER_COUNT || decode.stack.ownerCaches.size !== 15) {
    throw new Error("Gemma fixed prefill requires the complete 35-layer decode model");
  }
  const cacheCapacity = Math.max(
    ...Array.from(decode.stack.ownerCaches.values(), ({ capacity }) => capacity),
  );
  if (cacheCapacity < GEMMA_FIXED_PREFILL_ROWS) {
    throw new Error(`Gemma fixed prefill requires cache capacity ${GEMMA_FIXED_PREFILL_ROWS}`);
  }

  const ownedBuffers: GPUBuffer[] = [];
  const parameterArena = new GemmaPrefillParameterArena(device);
  const own = <T extends { ownedBuffers: GPUBuffer[] }>(resources: T): T => {
    ownedBuffers.push(...resources.ownedBuffers);
    return resources;
  };
  const allocate = (label: string, elements: number): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: elements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    ownedBuffers.push(buffer);
    return buffer;
  };

  const input = createGemmaDecodeInputResources(
    device,
    decode.inputPipeline,
    null,
    GEMMA_FIXED_PREFILL_ROWS,
    decode.input.modelWeights,
  );
  ownedBuffers.push(...input.buffers);
  const scratch: GemmaPrefillScratch = {
    hiddenNorm: allocate("Gemma prefill hidden norm", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
    residualNorm: allocate("Gemma prefill residual norm", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
    query: allocate("Gemma prefill query", GEMMA_FIXED_PREFILL_ROWS * MAX_HEAD_OUTPUT),
    queryNorm: allocate("Gemma prefill normalized query", GEMMA_FIXED_PREFILL_ROWS * MAX_HEAD_OUTPUT),
    key: allocate("Gemma prefill key", GEMMA_FIXED_PREFILL_ROWS * 512),
    keyNorm: allocate("Gemma prefill normalized key", GEMMA_FIXED_PREFILL_ROWS * 512),
    value: allocate("Gemma prefill value", GEMMA_FIXED_PREFILL_ROWS * 512),
    valueNorm: allocate("Gemma prefill normalized value", GEMMA_FIXED_PREFILL_ROWS * 512),
    attention: allocate("Gemma prefill attention", GEMMA_FIXED_PREFILL_ROWS * MAX_HEAD_OUTPUT),
    projection: allocate("Gemma prefill projection", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
    gate: allocate("Gemma prefill gate", GEMMA_FIXED_PREFILL_ROWS * MAX_INTERMEDIATE),
    up: allocate("Gemma prefill up", GEMMA_FIXED_PREFILL_ROWS * MAX_INTERMEDIATE),
    activated: allocate("Gemma prefill activated gate", GEMMA_FIXED_PREFILL_ROWS * MAX_INTERMEDIATE),
    down: allocate("Gemma prefill down", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
    pleInput: allocate("Gemma prefill PLE input", GEMMA_FIXED_PREFILL_ROWS * PLE_SIZE),
    pleGate: allocate("Gemma prefill PLE gate", GEMMA_FIXED_PREFILL_ROWS * PLE_SIZE),
    pleActivated: allocate("Gemma prefill PLE activated", GEMMA_FIXED_PREFILL_ROWS * PLE_SIZE),
    pleProjection: allocate("Gemma prefill PLE projection", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
    srq: allocate("Gemma prefill shared SRQ", GEMMA_FIXED_PREFILL_ROWS * MAX_INTERMEDIATE),
    finalNorm: allocate("Gemma prefill final norm", GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE),
  };

  const [stridedCopyPipeline, elementwisePipelines] = await Promise.all([
    getGemmaPrefillStridedCopyPipeline(device),
    getGemmaPrefillElementwisePipelines(device),
  ]);
  const layers: GemmaPrefillLayerResources[] = [];

  try {
    for (let layerIndex = 0; layerIndex < LAYER_COUNT; layerIndex += 1) {
      const decodeLayer = decode.stack.layers[layerIndex];
      const attentionWeights = decodeLayer.attention.modelWeights;
      const attentionScales = decodeLayer.attention.modelScales;
      const mlpWeights = decodeLayer.mlp.modelWeights;
      const mlpScales = decodeLayer.mlp.modelScales;
      const headDimension = decode.stack.pipelines[layerIndex].attention.headDim;
      const queryFeatures = decode.stack.pipelines[layerIndex].attention.qOutFeatures;
      const kvFeatures = decode.stack.pipelines[layerIndex].attention.kvOutFeatures;
      const mlpBits = decode.stack.pipelines[layerIndex].mlp.bitWidth;
      const intermediateFeatures = decode.stack.pipelines[layerIndex].mlp.intermediateFeatures;
      const ownsKv = decodeLayer.attention.writesKvCache;
      const cache = decodeLayer.attention.cache;
      const window = headDimension === 256 ? 512 : 0;

      const norm = async (
        dimension: number,
        weighted: boolean,
        source: GPUBuffer,
        weight: GemmaPrefillRmsBufferSlice | null,
        output: GPUBuffer,
        rows = GEMMA_FIXED_PREFILL_ROWS,
      ): Promise<GemmaPrefillNorm> => {
        const pipeline = await getGemmaPrefillRmsPipeline(device, dimension, weighted);
        return {
          pipeline,
          resources: own(createGemmaPrefillRmsResources(
            device,
            pipeline,
            rows,
            source,
            weight,
            output,
            parameterArena,
          )),
        };
      };
      const normResidual = async (
        source: GPUBuffer,
        weight: GemmaPrefillRmsBufferSlice,
        residual: GPUBuffer,
        factors: GPUBuffer | null = null,
        factorIndex = 0,
      ): Promise<GemmaPrefillNormResidual> => {
        const pipeline = await getGemmaPrefillRmsResidualPipeline(
          device,
          HIDDEN_SIZE,
          factors !== null,
        );
        return {
          pipeline,
          resources: own(createGemmaPrefillRmsResidualResources(
            device,
            pipeline,
            GEMMA_FIXED_PREFILL_ROWS,
            source,
            weight,
            residual,
            factors,
            factorIndex,
            parameterArena,
          )),
        };
      };
      const projection = async (
        inFeatures: number,
        outFeatures: number,
        bits: 2 | 4,
        source: GPUBuffer,
        weights: GemmaPrefillQatLinearWeights,
        output: GPUBuffer,
      ): Promise<GemmaPrefillProjection> => {
        const pipelines = await getGemmaPrefillQatLinearPipelines(device, {
          rows: GEMMA_FIXED_PREFILL_ROWS,
          inFeatures,
          outFeatures,
          bits,
        });
        return {
          pipelines,
          resources: own(createGemmaPrefillQatLinearResources(
            device,
            pipelines,
            source,
            weights,
            output,
            scratch.srq,
            parameterArena,
          )),
        };
      };
      const rope = async (
        activations: GPUBuffer,
        heads: number,
      ): Promise<GemmaPrefillRope> => {
        const pipeline = await getGemmaPrefillRopePipeline(device, headDimension);
        const rotary = headDimension === 256
          ? emptyRotary(GEMMA_FIXED_PREFILL_ROWS, 128)
          : emptyRotary(GEMMA_FIXED_PREFILL_ROWS, 256);
        return {
          pipeline,
          resources: own(createGemmaPrefillRopeResources(
            device,
            pipeline,
            activations,
            GEMMA_FIXED_PREFILL_ROWS,
            heads,
            rotary,
            parameterArena,
          )),
        };
      };
      const pleProjection = async (
        inFeatures: 256 | 1536,
        outFeatures: 256 | 1536,
        source: GPUBuffer,
        codes: GPUBuffer,
        rowScales: GPUBuffer,
        inputScale: number,
        outputScale: number,
        output: GPUBuffer,
      ): Promise<GemmaPrefillPleProjection> => {
        const pipeline = await getGemmaPrefillPleDensePipeline(device, {
          rows: GEMMA_FIXED_PREFILL_ROWS,
          inFeatures,
          outFeatures,
        });
        return {
          pipeline,
          resources: own(createGemmaPrefillPleDenseResources(
            device,
            pipeline,
            source,
            { codes, rowScales, inputScale, outputScale },
            output,
            parameterArena,
          )),
        };
      };

      const inputNormWeight = layerIndex === 0
        ? requiredWeight(attentionWeights.inputNorm, "layer-0 input norm")
        : slice(
            decode.stack.layers[layerIndex - 1].mlp.modelWeights
              .postPleNextInputNormAndLayerScale,
            HIDDEN_BYTES,
            HIDDEN_BYTES,
          );
      const inputNorm = await norm(
        HIDDEN_SIZE,
        true,
        input.hidden,
        inputNormWeight,
        scratch.hiddenNorm,
      );

      const attentionBits: 4 = 4;
      const packedRowBytes = HIDDEN_SIZE * attentionBits / 8;
      const queryPackedBytes = queryFeatures * packedRowBytes;
      const kvPackedBytes = kvFeatures * packedRowBytes;
      const query = await projection(
        HIDDEN_SIZE,
        queryFeatures,
        attentionBits,
        scratch.hiddenNorm,
        {
          packedWeights: slice(attentionWeights.qkvPacked, 0, queryPackedBytes),
          rowScales: slice(attentionWeights.qkvRowScales, 0, queryFeatures * 4),
          inputScale: attentionScales.qkvInput,
          outputScale: attentionScales.qkvOutput[0],
        },
        scratch.query,
      );
      const queryNorm = await norm(
        headDimension,
        true,
        scratch.query,
        attentionWeights.qNorm,
        scratch.queryNorm,
        GEMMA_FIXED_PREFILL_ROWS * QUERY_HEADS,
      );
      const queryRope = await rope(scratch.queryNorm, QUERY_HEADS);

      let key: GemmaPrefillProjection | null = null;
      let keyNorm: GemmaPrefillNorm | null = null;
      let keyRope: GemmaPrefillRope | null = null;
      let value: GemmaPrefillProjection | null = null;
      let valueNorm: GemmaPrefillNorm | null = null;
      let keyCopy: GemmaPrefillStridedCopyResources | null = null;
      let valueCopy: GemmaPrefillStridedCopyResources | null = null;
      if (ownsKv) {
        key = await projection(
          HIDDEN_SIZE,
          kvFeatures,
          attentionBits,
          qkvSrqMode === "shared" ? scratch.srq : scratch.hiddenNorm,
          {
            packedWeights: slice(
              attentionWeights.qkvPacked,
              queryPackedBytes,
              kvPackedBytes,
            ),
            rowScales: slice(
              attentionWeights.qkvRowScales,
              queryFeatures * 4,
              kvFeatures * 4,
            ),
            inputScale: qkvSrqMode === "shared" ? 0 : attentionScales.qkvInput,
            outputScale: attentionScales.qkvOutput[1],
          },
          scratch.key,
        );
        keyNorm = await norm(
          headDimension,
          true,
          scratch.key,
          requiredWeight(attentionWeights.kNorm, `layer ${layerIndex} K norm`),
          scratch.keyNorm,
        );
        keyRope = await rope(scratch.keyNorm, 1);
        value = await projection(
          HIDDEN_SIZE,
          kvFeatures,
          attentionBits,
          qkvSrqMode === "shared" ? scratch.srq : scratch.hiddenNorm,
          {
            packedWeights: slice(
              attentionWeights.qkvPacked,
              queryPackedBytes + kvPackedBytes,
              kvPackedBytes,
            ),
            rowScales: slice(
              attentionWeights.qkvRowScales,
              (queryFeatures + kvFeatures) * 4,
              kvFeatures * 4,
            ),
            inputScale: qkvSrqMode === "shared" ? 0 : attentionScales.qkvInput,
            outputScale: attentionScales.qkvOutput[2],
          },
          scratch.value,
        );
        valueNorm = await norm(
          headDimension,
          false,
          scratch.value,
          null,
          scratch.valueNorm,
        );
        keyCopy = own(createGemmaPrefillStridedCopyResources(
          device,
          stridedCopyPipeline,
          scratch.keyNorm,
          cache.keyBuffer,
          copyParameters(kvFeatures),
          parameterArena,
        ));
        valueCopy = own(createGemmaPrefillStridedCopyResources(
          device,
          stridedCopyPipeline,
          scratch.valueNorm,
          cache.valueBuffer,
          copyParameters(kvFeatures),
          parameterArena,
        ));
      }

      const attentionPipeline = await getGemmaPrefillAttentionPipeline(device, headDimension);
      const attention = own(createGemmaPrefillAttentionResources(
        device,
        attentionPipeline,
        scratch.queryNorm,
        cache.keyBuffer,
        cache.valueBuffer,
        GEMMA_FIXED_PREFILL_ROWS,
        cache.capacity,
        {
          sequence: GEMMA_FIXED_PREFILL_ROWS,
          keyLength: GEMMA_FIXED_PREFILL_ROWS,
          queryOffset: 0,
          queryHeads: QUERY_HEADS,
          kvHeads: 1,
          window,
        },
        scratch.attention,
        parameterArena,
      ));
      const outputProjection = await projection(
        queryFeatures,
        HIDDEN_SIZE,
        attentionBits,
        scratch.attention,
        {
          packedWeights: attentionWeights.outputProjectionPacked,
          rowScales: attentionWeights.outputProjectionRowScales,
          inputScale: attentionScales.attentionOutput,
          outputScale: attentionScales.outputProjectionOutput,
        },
        scratch.projection,
      );
      const postAttentionWeight = slice(
        attentionWeights.postAttentionAndPreFeedforwardNorm,
        0,
        HIDDEN_BYTES,
      );
      const postAttentionNorm = rmsEpilogueMode === "separate"
        ? await norm(
            HIDDEN_SIZE,
            true,
            scratch.projection,
            postAttentionWeight,
            scratch.residualNorm,
          )
        : null;
      const attentionResidual = rmsEpilogueMode === "separate"
        ? own(createGemmaPrefillAddResources(
            device,
            elementwisePipelines.add,
            input.hidden,
            scratch.residualNorm,
            GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
            parameterArena,
          ))
        : null;
      const attentionNormResidual = rmsEpilogueMode === "fused"
        ? await normResidual(scratch.projection, postAttentionWeight, input.hidden)
        : null;
      const preFeedforwardNorm = await norm(
        HIDDEN_SIZE,
        true,
        input.hidden,
        slice(
          attentionWeights.postAttentionAndPreFeedforwardNorm,
          HIDDEN_BYTES,
          HIDDEN_BYTES,
        ),
        scratch.hiddenNorm,
      );
      const gateWeights: GemmaPrefillQatLinearWeights = {
        packedWeights: mlpWeights.gatePacked,
        rowScales: mlpWeights.gateRowScales,
        inputScale: mlpScales.gateInput,
        outputScale: mlpScales.gateOutput,
      };
      const upWeights: GemmaPrefillQatLinearWeights = {
        packedWeights: mlpWeights.upPacked,
        rowScales: mlpWeights.upRowScales,
        inputScale: mlpScales.gateInput,
        outputScale: mlpScales.upOutput,
      };
      let gateUpActivation: GemmaPrefillGateUpActivationProjection | null = null;
      let gateUp: GemmaPrefillGateUpProjection | null = null;
      let gate: GemmaPrefillProjection | null = null;
      let up: GemmaPrefillProjection | null = null;
      if (gateUpMode === "fused-activated") {
        const gateUpPipelines = await getGemmaPrefillQatGateUpActivationPipelines(device, {
          rows: GEMMA_FIXED_PREFILL_ROWS,
          inFeatures: HIDDEN_SIZE,
          outFeatures: intermediateFeatures,
          bits: mlpBits,
        });
        gateUpActivation = {
          pipelines: gateUpPipelines,
          resources: own(createGemmaPrefillQatGateUpActivationResources(
            device,
            gateUpPipelines,
            scratch.hiddenNorm,
            gateWeights,
            upWeights,
            mlpWeights.gateGeluLut,
            mlpScales.downInput,
            scratch.srq,
            scratch.activated,
            parameterArena,
          )),
        };
      } else if (gateUpMode === "fused") {
        const gateUpPipelines = await getGemmaPrefillQatGateUpPipelines(device, {
          rows: GEMMA_FIXED_PREFILL_ROWS,
          inFeatures: HIDDEN_SIZE,
          outFeatures: intermediateFeatures,
          bits: mlpBits,
        });
        gateUp = {
          pipelines: gateUpPipelines,
          resources: own(createGemmaPrefillQatGateUpResources(
            device,
            gateUpPipelines,
            scratch.hiddenNorm,
            gateWeights,
            upWeights,
            scratch.gate,
            scratch.up,
            scratch.srq,
            parameterArena,
          )),
        };
      } else {
        gate = await projection(
          HIDDEN_SIZE,
          intermediateFeatures,
          mlpBits,
          scratch.hiddenNorm,
          gateWeights,
          scratch.gate,
        );
        up = await projection(
          HIDDEN_SIZE,
          intermediateFeatures,
          mlpBits,
          scratch.hiddenNorm,
          upWeights,
          scratch.up,
        );
      }
      const gateActivation = gateUpActivation ? null : own(
        createGemmaPrefillGeluMultiplyResources(
          device,
          elementwisePipelines.geluMultiply,
          scratch.gate,
          scratch.up,
          mlpWeights.gateGeluLut,
          scratch.activated,
          GEMMA_FIXED_PREFILL_ROWS * intermediateFeatures,
          mlpScales.gateOutput,
          parameterArena,
        ),
      );
      const down = await projection(
        intermediateFeatures,
        HIDDEN_SIZE,
        mlpBits,
        gateUpActivation ? scratch.srq : scratch.activated,
        {
          packedWeights: mlpWeights.downPacked,
          rowScales: mlpWeights.downRowScales,
          inputScale: gateUpActivation ? 0 : mlpScales.downInput,
          outputScale: mlpScales.downOutput,
        },
        scratch.down,
      );
      const postFeedforwardNorm = rmsEpilogueMode === "separate"
        ? await norm(
            HIDDEN_SIZE,
            true,
            scratch.down,
            mlpWeights.postFeedforwardNorm,
            scratch.residualNorm,
          )
        : null;
      const feedforwardResidual = rmsEpilogueMode === "separate"
        ? own(createGemmaPrefillAddResources(
            device,
            elementwisePipelines.add,
            input.hidden,
            scratch.residualNorm,
            GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
            parameterArena,
          ))
        : null;
      const feedforwardNormResidual = rmsEpilogueMode === "fused"
        ? await normResidual(scratch.down, mlpWeights.postFeedforwardNorm, input.hidden)
        : null;
      const pleInputCopy = pleInputMode === "copied"
        ? own(createGemmaPrefillStridedCopyResources(
            device,
            stridedCopyPipeline,
            input.perLayerInputs,
            scratch.pleInput,
            {
              rows: GEMMA_FIXED_PREFILL_ROWS,
              sourceStride: LAYER_COUNT * PLE_SIZE,
              sourceStart: layerIndex * PLE_SIZE,
              destinationStride: PLE_SIZE,
              destinationStart: 0,
              copyColumns: PLE_SIZE,
            },
            parameterArena,
          ))
        : null;
      const pleGate = await pleProjection(
        HIDDEN_SIZE,
        PLE_SIZE,
        input.hidden,
        mlpWeights.pleGatePacked,
        mlpWeights.pleGateRowScales,
        mlpScales.pleGateInput,
        mlpScales.pleGateOutput,
        scratch.pleGate,
      );
      const pleActivation = pleInputMode === "direct"
        ? own(createGemmaPrefillStridedGeluMultiplyResources(
            device,
            elementwisePipelines.geluMultiplyStrided,
            scratch.pleGate,
            input.perLayerInputs,
            mlpWeights.pleGeluLut,
            scratch.pleActivated,
            GEMMA_FIXED_PREFILL_ROWS,
            PLE_SIZE,
            LAYER_COUNT * PLE_SIZE,
            layerIndex * PLE_SIZE,
            mlpScales.pleGateOutput,
            parameterArena,
          ))
        : own(createGemmaPrefillGeluMultiplyResources(
            device,
            elementwisePipelines.geluMultiply,
            scratch.pleGate,
            scratch.pleInput,
            mlpWeights.pleGeluLut,
            scratch.pleActivated,
            GEMMA_FIXED_PREFILL_ROWS * PLE_SIZE,
            mlpScales.pleGateOutput,
            parameterArena,
          ));
      const pleProjectionResult = await pleProjection(
        PLE_SIZE,
        HIDDEN_SIZE,
        scratch.pleActivated,
        mlpWeights.pleProjectionPacked,
        mlpWeights.pleProjectionRowScales,
        mlpScales.pleProjectionInput,
        mlpScales.pleProjectionOutput,
        scratch.pleProjection,
      );
      const postPleWeight = slice(
        mlpWeights.postPleNextInputNormAndLayerScale,
        0,
        HIDDEN_BYTES,
      );
      const postPleNorm = rmsEpilogueMode === "separate"
        ? await norm(
            HIDDEN_SIZE,
            true,
            scratch.pleProjection,
            postPleWeight,
            scratch.residualNorm,
          )
        : null;
      const pleResidual = rmsEpilogueMode === "separate"
        ? own(createGemmaPrefillAddResources(
            device,
            elementwisePipelines.add,
            input.hidden,
            scratch.residualNorm,
            GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
            parameterArena,
          ))
        : null;
      const layerScale = rmsEpilogueMode === "separate"
        ? own(createGemmaPrefillMultiplyResources(
            device,
            elementwisePipelines.multiply,
            input.hidden,
            mlpWeights.postPleNextInputNormAndLayerScale,
            GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
            HIDDEN_SIZE * 2,
            parameterArena,
          ))
        : null;
      const pleNormResidualScale = rmsEpilogueMode === "fused"
        ? await normResidual(
            scratch.pleProjection,
            postPleWeight,
            input.hidden,
            mlpWeights.postPleNextInputNormAndLayerScale,
            HIDDEN_SIZE * 2,
          )
        : null;

      layers.push({
        layerIndex,
        headDimension,
        kvFeatures,
        window,
        cacheCapacity: cache.capacity,
        inputNorm,
        query,
        queryNorm,
        queryRope,
        key,
        keyNorm,
        keyRope,
        value,
        valueNorm,
        keyCopy,
        valueCopy,
        attentionPipeline,
        attention,
        outputProjection,
        postAttentionNorm,
        attentionResidual,
        attentionNormResidual,
        preFeedforwardNorm,
        gateUpActivation,
        gateUp,
        gate,
        up,
        gateActivation,
        down,
        postFeedforwardNorm,
        feedforwardResidual,
        feedforwardNormResidual,
        pleInputCopy,
        pleGate,
        pleActivation,
        pleProjection: pleProjectionResult,
        postPleNorm,
        pleResidual,
        layerScale,
        pleNormResidualScale,
      });
    }

    const finalNormPipeline = await getGemmaPrefillRmsPipeline(device, HIDDEN_SIZE, true);
    const finalNorm = {
      pipeline: finalNormPipeline,
      resources: own(createGemmaPrefillRmsResources(
        device,
        finalNormPipeline,
        GEMMA_FIXED_PREFILL_ROWS,
        input.hidden,
        slice(
          decode.stack.layers[LAYER_COUNT - 1].mlp.modelWeights
            .postPleNextInputNormAndLayerScale,
          HIDDEN_BYTES,
          HIDDEN_BYTES,
        ),
        scratch.finalNorm,
        parameterArena,
      )),
    };
    const lastRow = allocate("Gemma prefill actual last row", HIDDEN_SIZE);
    const lastRowCopy = own(createGemmaPrefillStridedCopyResources(
      device,
      stridedCopyPipeline,
      scratch.finalNorm,
      lastRow,
      {
        rows: 1,
        sourceStride: HIDDEN_SIZE,
        sourceStart: 0,
        destinationStride: HIDDEN_SIZE,
        destinationStart: 0,
        copyColumns: HIDDEN_SIZE,
      },
      parameterArena,
    ));
    const lmHeadPipelines = await getGemmaPrefillQatLinearPipelines(device, {
      rows: 1,
      inFeatures: HIDDEN_SIZE,
      outFeatures: VOCAB_SIZE,
      bits: 2,
    });
    const lmHead = {
      pipelines: lmHeadPipelines,
      resources: own(createGemmaPrefillQatLinearResources(
        device,
        lmHeadPipelines,
        lastRow,
        {
          packedWeights: decode.lmHead.modelWeights.packed,
          rowScales: decode.lmHead.modelWeights.rowScales,
          inputScale: decode.lmHead.modelWeights.inputScale,
          outputScale: decode.lmHead.modelWeights.outputScale,
        },
        decode.logits,
        scratch.srq,
        parameterArena,
      )),
    };
    ownedBuffers.push(...parameterArena.buffers);
    return {
      input,
      layers,
      finalNorm,
      lastRow,
      lastRowCopy,
      lmHead,
      stridedCopyPipeline,
      elementwisePipelines,
      decode,
      cacheCapacity,
      ownedBuffers,
    };
  } catch (error) {
    for (const buffer of ownedBuffers.toReversed()) buffer.destroy();
    parameterArena.destroy();
    throw error;
  }
}

export function updateGemmaFixedPrefill(
  device: GPUDevice,
  resources: GemmaFixedPrefillResources,
  position: number,
  validRows: number,
  rotary: GemmaRotaryBlock,
): void {
  if (!Number.isInteger(position) || position < 0 ||
      !Number.isInteger(validRows) || validRows < 1 || validRows > GEMMA_FIXED_PREFILL_ROWS ||
      position + GEMMA_FIXED_PREFILL_ROWS > resources.cacheCapacity ||
      rotary.rowCount !== GEMMA_FIXED_PREFILL_ROWS) {
    throw new Error("Gemma fixed prefill runtime geometry is invalid");
  }
  for (const [layerIndex, cache] of resources.decode.stack.ownerCaches) {
    if (cache.length !== position) {
      throw new Error(
        `Gemma owner cache ${layerIndex} length ${cache.length} does not match prefill position ${position}`,
      );
    }
  }
  for (const layer of resources.layers) {
    const layerRotary = layer.headDimension === 256 ? rotary.sliding : rotary.full;
    updateGemmaPrefillRope(
      device,
      layer.queryRope.pipeline,
      layer.queryRope.resources,
      GEMMA_FIXED_PREFILL_ROWS,
      layerRotary,
    );
    if (layer.keyRope) {
      updateGemmaPrefillRope(
        device,
        layer.keyRope.pipeline,
        layer.keyRope.resources,
        GEMMA_FIXED_PREFILL_ROWS,
        layerRotary,
      );
    }
    if (layer.keyCopy && layer.valueCopy) {
      const cachePosition = position % layer.cacheCapacity;
      if (cachePosition + GEMMA_FIXED_PREFILL_ROWS > layer.cacheCapacity) {
        throw new Error("Gemma fixed prefill block crosses a physical cache boundary");
      }
      const parameters = copyParameters(layer.kvFeatures, cachePosition * layer.kvFeatures);
      updateGemmaPrefillStridedCopy(device, layer.keyCopy, parameters);
      updateGemmaPrefillStridedCopy(device, layer.valueCopy, parameters);
    }
    updateGemmaPrefillAttention(
      device,
      layer.attention,
      resources.cacheCapacity,
      {
        sequence: GEMMA_FIXED_PREFILL_ROWS,
        keyLength: position + GEMMA_FIXED_PREFILL_ROWS,
        queryOffset: position,
        queryHeads: QUERY_HEADS,
        kvHeads: 1,
        window: layer.window,
      },
    );
  }
  updateGemmaPrefillStridedCopy(device, resources.lastRowCopy, {
    rows: 1,
    sourceStride: HIDDEN_SIZE,
    sourceStart: (validRows - 1) * HIDDEN_SIZE,
    destinationStride: HIDDEN_SIZE,
    destinationStart: 0,
    copyColumns: HIDDEN_SIZE,
  });
}

export function encodeGemmaFixedPrefill(
  encoder: GPUCommandEncoder,
  resources: GemmaFixedPrefillResources,
  outputMode: GemmaModelOutputMode = "greedy",
  logitsReadback?: GPUBuffer,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma fixed-32 prefill" });
  encodeGemmaDecodeInputPass(pass, resources.decode.inputPipeline, resources.input);
  for (const layer of resources.layers) {
    encodeLayerAttentionStage(pass, resources, layer);
    encodeLayerFeedforwardStage(pass, resources, layer);
    encodeLayerPleStage(pass, resources, layer);
  }
  if (outputMode !== "none") {
    encodeOutputStage(pass, resources, outputMode);
  }
  pass.end();
  encodeOutputReadback(encoder, resources, outputMode, logitsReadback);
}

function encodeGemmaFixedPrefillProfiled(
  encoder: GPUCommandEncoder,
  resources: GemmaFixedPrefillResources,
  outputMode: GemmaModelOutputMode,
  logitsReadback: GPUBuffer | undefined,
  profile: GemmaFixedPrefillProfileResources,
): void {
  const encodeStage = (
    label: string,
    stage: GemmaFixedPrefillGpuStage,
    layerIndex: number | null,
    encode: (pass: GPUComputePassEncoder) => void,
  ) => {
    const beginningOfPassWriteIndex = profile.records.length * 2;
    const pass = encoder.beginComputePass({
      label,
      timestampWrites: {
        querySet: profile.querySet,
        beginningOfPassWriteIndex,
        endOfPassWriteIndex: beginningOfPassWriteIndex + 1,
      },
    });
    encode(pass);
    pass.end();
    profile.records.push({ stage, layerIndex });
  };

  encodeStage("Gemma fixed prefill input", "input", null, (pass) => {
    encodeGemmaDecodeInputPass(pass, resources.decode.inputPipeline, resources.input);
  });
  for (const layer of resources.layers) {
    encodeStage(
      `Gemma fixed prefill layer ${layer.layerIndex} attention`,
      "attention",
      layer.layerIndex,
      (pass) => encodeLayerAttentionStage(pass, resources, layer),
    );
    encodeStage(
      `Gemma fixed prefill layer ${layer.layerIndex} feedforward`,
      "feedforward",
      layer.layerIndex,
      (pass) => encodeLayerFeedforwardStage(pass, resources, layer),
    );
    encodeStage(
      `Gemma fixed prefill layer ${layer.layerIndex} PLE`,
      "ple",
      layer.layerIndex,
      (pass) => encodeLayerPleStage(pass, resources, layer),
    );
  }
  if (outputMode !== "none") {
    encodeStage("Gemma fixed prefill output", "output", null, (pass) => {
      encodeOutputStage(pass, resources, outputMode);
    });
  }
  encodeOutputReadback(encoder, resources, outputMode, logitsReadback);
  encoder.resolveQuerySet(
    profile.querySet,
    0,
    profile.records.length * 2,
    profile.resolveBuffer,
    0,
  );
  encoder.copyBufferToBuffer(
    profile.resolveBuffer,
    0,
    profile.readBuffer,
    0,
    profile.records.length * 16,
  );
}

function encodeOutputReadback(
  encoder: GPUCommandEncoder,
  resources: GemmaFixedPrefillResources,
  outputMode: GemmaModelOutputMode,
  logitsReadback?: GPUBuffer,
): void {
  if (outputMode === "greedy") {
    encoder.copyBufferToBuffer(
      resources.decode.greedy.result,
      0,
      resources.decode.greedy.readback,
      0,
      8,
    );
  } else if (outputMode === "logits") {
    if (!logitsReadback || logitsReadback.size < resources.decode.logits.size) {
      throw new Error("Gemma prefill logits output requires a matching readback buffer");
    }
    encoder.copyBufferToBuffer(
      resources.decode.logits,
      0,
      logitsReadback,
      0,
      resources.decode.logits.size,
    );
  }
}

export async function submitGemmaFixedPrefill(
  device: GPUDevice,
  resources: GemmaFixedPrefillResources,
  position: number,
  validRows: number,
  outputMode: GemmaModelOutputMode = "greedy",
  logitsReadback?: GPUBuffer,
  profileGpuStages = false,
): Promise<GemmaFixedPrefillSubmission> {
  const encoder = device.createCommandEncoder({ label: "Gemma fixed-32 prefill" });
  const profile = profileGpuStages && device.features.has("timestamp-query")
    ? createProfileResources(device, outputMode)
    : null;
  if (profile) {
    encodeGemmaFixedPrefillProfiled(
      encoder,
      resources,
      outputMode,
      logitsReadback,
      profile,
    );
  } else {
    encodeGemmaFixedPrefill(encoder, resources, outputMode, logitsReadback);
  }
  device.queue.submit([encoder.finish()]);
  for (const cache of resources.decode.stack.ownerCaches.values()) {
    cache.commitWrite(position, validRows);
  }
  try {
    let output: GemmaModelOutput;
    if (outputMode === "greedy") {
      output = {
        prediction: await readGemmaGreedyResult(resources.decode.greedy),
        logits: null,
        logitsReadbackMs: 0,
      };
    } else if (outputMode === "logits") {
      const startedAt = performance.now();
      await logitsReadback!.mapAsync(GPUMapMode.READ);
      const logits = new Float32Array(
        logitsReadback!.getMappedRange(0, resources.decode.logits.size).slice(0),
      );
      logitsReadback!.unmap();
      output = {
        prediction: null,
        logits,
        logitsReadbackMs: performance.now() - startedAt,
      };
    } else {
      output = { prediction: null, logits: null, logitsReadbackMs: 0 };
    }
    return {
      ...output,
      gpuProfile: profile
        ? await readProfile(profile, position, validRows)
        : null,
    };
  } finally {
    destroyProfileResources(profile);
  }
}

export function destroyGemmaFixedPrefillResources(
  resources: GemmaFixedPrefillResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

function encodeNorm(pass: GPUComputePassEncoder, norm: GemmaPrefillNorm): void {
  encodeGemmaPrefillRmsPass(pass, norm.pipeline, norm.resources);
}

function encodeLayerAttentionStage(
  pass: GPUComputePassEncoder,
  resources: GemmaFixedPrefillResources,
  layer: GemmaPrefillLayerResources,
): void {
  encodeNorm(pass, layer.inputNorm);
  encodeProjection(pass, layer.query);
  encodeNorm(pass, layer.queryNorm);
  encodeRope(pass, layer.queryRope);
  if (layer.key && layer.keyNorm && layer.keyRope && layer.value && layer.valueNorm &&
      layer.keyCopy && layer.valueCopy) {
    encodeProjection(pass, layer.key);
    encodeNorm(pass, layer.keyNorm);
    encodeRope(pass, layer.keyRope);
    encodeProjection(pass, layer.value);
    encodeNorm(pass, layer.valueNorm);
    encodeGemmaPrefillStridedCopyPass(
      pass,
      resources.stridedCopyPipeline,
      layer.keyCopy,
      GEMMA_FIXED_PREFILL_ROWS,
    );
    encodeGemmaPrefillStridedCopyPass(
      pass,
      resources.stridedCopyPipeline,
      layer.valueCopy,
      GEMMA_FIXED_PREFILL_ROWS,
    );
  }
  encodeGemmaPrefillAttentionPass(
    pass,
    layer.attentionPipeline,
    layer.attention,
    GEMMA_FIXED_PREFILL_ROWS,
  );
  encodeProjection(pass, layer.outputProjection);
  if (layer.attentionNormResidual) {
    encodeGemmaPrefillRmsResidualPass(
      pass,
      layer.attentionNormResidual.pipeline,
      layer.attentionNormResidual.resources,
    );
  } else if (layer.postAttentionNorm && layer.attentionResidual) {
    encodeNorm(pass, layer.postAttentionNorm);
    encodeGemmaPrefillElementwisePass(
      pass,
      resources.elementwisePipelines.add,
      layer.attentionResidual,
    );
  } else {
    throw new Error("Gemma prefill layer has no attention RMS epilogue");
  }
}

function encodeLayerFeedforwardStage(
  pass: GPUComputePassEncoder,
  resources: GemmaFixedPrefillResources,
  layer: GemmaPrefillLayerResources,
): void {
  encodeNorm(pass, layer.preFeedforwardNorm);
  if (layer.gateUpActivation) {
    encodeGemmaPrefillQatGateUpActivationPass(
      pass,
      layer.gateUpActivation.pipelines,
      layer.gateUpActivation.resources,
    );
  } else if (layer.gateUp) {
    encodeGemmaPrefillQatGateUpPass(
      pass,
      layer.gateUp.pipelines,
      layer.gateUp.resources,
    );
  } else if (layer.gate && layer.up) {
    encodeProjection(pass, layer.gate);
    encodeProjection(pass, layer.up);
  } else {
    throw new Error("Gemma prefill layer has no gate/up implementation");
  }
  if (layer.gateActivation) {
    encodeGemmaPrefillElementwisePass(
      pass,
      resources.elementwisePipelines.geluMultiply,
      layer.gateActivation,
    );
  }
  encodeProjection(pass, layer.down);
  if (layer.feedforwardNormResidual) {
    encodeGemmaPrefillRmsResidualPass(
      pass,
      layer.feedforwardNormResidual.pipeline,
      layer.feedforwardNormResidual.resources,
    );
  } else if (layer.postFeedforwardNorm && layer.feedforwardResidual) {
    encodeNorm(pass, layer.postFeedforwardNorm);
    encodeGemmaPrefillElementwisePass(
      pass,
      resources.elementwisePipelines.add,
      layer.feedforwardResidual,
    );
  } else {
    throw new Error("Gemma prefill layer has no feedforward RMS epilogue");
  }
}

function encodeLayerPleStage(
  pass: GPUComputePassEncoder,
  resources: GemmaFixedPrefillResources,
  layer: GemmaPrefillLayerResources,
): void {
  if (layer.pleInputCopy) {
    encodeGemmaPrefillStridedCopyPass(
      pass,
      resources.stridedCopyPipeline,
      layer.pleInputCopy,
      GEMMA_FIXED_PREFILL_ROWS,
    );
  }
  encodePleProjection(pass, layer.pleGate);
  encodeGemmaPrefillElementwisePass(
    pass,
    layer.pleInputCopy
      ? resources.elementwisePipelines.geluMultiply
      : resources.elementwisePipelines.geluMultiplyStrided,
    layer.pleActivation,
  );
  encodePleProjection(pass, layer.pleProjection);
  if (layer.pleNormResidualScale) {
    encodeGemmaPrefillRmsResidualPass(
      pass,
      layer.pleNormResidualScale.pipeline,
      layer.pleNormResidualScale.resources,
    );
  } else if (layer.postPleNorm && layer.pleResidual && layer.layerScale) {
    encodeNorm(pass, layer.postPleNorm);
    encodeGemmaPrefillElementwisePass(
      pass,
      resources.elementwisePipelines.add,
      layer.pleResidual,
    );
    encodeGemmaPrefillElementwisePass(
      pass,
      resources.elementwisePipelines.multiply,
      layer.layerScale,
    );
  } else {
    throw new Error("Gemma prefill layer has no PLE RMS epilogue");
  }
}

function encodeOutputStage(
  pass: GPUComputePassEncoder,
  resources: GemmaFixedPrefillResources,
  outputMode: GemmaModelOutputMode,
): void {
  encodeNorm(pass, resources.finalNorm);
  encodeGemmaPrefillStridedCopyPass(
    pass,
    resources.stridedCopyPipeline,
    resources.lastRowCopy,
    1,
  );
  encodeProjection(pass, resources.lmHead);
  if (outputMode === "greedy") {
    encodeGemmaGreedyPass(
      pass,
      resources.decode.greedyPipelines,
      resources.decode.greedy,
    );
  }
}

function createProfileResources(
  device: GPUDevice,
  outputMode: GemmaModelOutputMode,
): GemmaFixedPrefillProfileResources {
  const recordCount = 1 + LAYER_COUNT * 3 + (outputMode === "none" ? 0 : 1);
  const queryCount = recordCount * 2;
  const size = queryCount * 8;
  return {
    querySet: device.createQuerySet({ type: "timestamp", count: queryCount }),
    resolveBuffer: device.createBuffer({
      label: "Gemma fixed prefill profile resolve",
      size,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readBuffer: device.createBuffer({
      label: "Gemma fixed prefill profile readback",
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    records: [],
  };
}

async function readProfile(
  profile: GemmaFixedPrefillProfileResources,
  position: number,
  validRows: number,
): Promise<GemmaFixedPrefillGpuProfile> {
  const byteLength = profile.records.length * 16;
  await profile.readBuffer.mapAsync(GPUMapMode.READ, 0, byteLength);
  const timestamps = new BigUint64Array(
    profile.readBuffer.getMappedRange(0, byteLength).slice(0),
  );
  profile.readBuffer.unmap();
  const stageGpuMs: Record<GemmaFixedPrefillGpuStage, number> = {
    input: 0,
    attention: 0,
    feedforward: 0,
    ple: 0,
    output: 0,
  };
  const samples = profile.records.map((record, index) => {
    const gpuMs = Number(timestamps[index * 2 + 1] - timestamps[index * 2]) / 1e6;
    stageGpuMs[record.stage] += gpuMs;
    return { ...record, gpuMs };
  });
  return {
    position,
    validRows,
    samples,
    stageGpuMs,
    totalGpuMs: samples.reduce((total, sample) => total + sample.gpuMs, 0),
  };
}

function destroyProfileResources(profile: GemmaFixedPrefillProfileResources | null): void {
  if (!profile) return;
  profile.querySet.destroy();
  profile.resolveBuffer.destroy();
  profile.readBuffer.destroy();
}

function encodeProjection(
  pass: GPUComputePassEncoder,
  projection: GemmaPrefillProjection,
): void {
  encodeGemmaPrefillQatLinearPass(pass, projection.pipelines, projection.resources);
}

function encodeRope(pass: GPUComputePassEncoder, rope: GemmaPrefillRope): void {
  encodeGemmaPrefillRopePass(
    pass,
    rope.pipeline,
    rope.resources,
    GEMMA_FIXED_PREFILL_ROWS,
  );
}

function encodePleProjection(
  pass: GPUComputePassEncoder,
  projection: GemmaPrefillPleProjection,
): void {
  encodeGemmaPrefillPleDensePass(pass, projection.pipeline, projection.resources);
}

function slice(buffer: GPUBuffer, offset: number, size: number): GemmaPrefillBufferSlice {
  return { buffer, offset, size };
}

function requiredWeight(buffer: GPUBuffer | null, label: string): GPUBuffer {
  if (!buffer) throw new Error(`Gemma fixed prefill is missing ${label}`);
  return buffer;
}

function copyParameters(kvFeatures: number, destinationStart = 0) {
  return {
    rows: GEMMA_FIXED_PREFILL_ROWS,
    sourceStride: kvFeatures,
    sourceStart: 0,
    destinationStride: kvFeatures,
    destinationStart,
    copyColumns: kvFeatures,
  };
}

function emptyRotary(rows: number, halfDimension: number) {
  return {
    cosine: new Float32Array(rows * halfDimension),
    sine: new Float32Array(rows * halfDimension),
  };
}
