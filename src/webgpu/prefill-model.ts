import type { GemmaRotaryBlock } from "../model/gemma-rope";
import {
  createGemmaDecodeInputResources,
  encodeGemmaDecodeInput,
  type GemmaDecodeInputResources,
} from "./decode-input";
import {
  encodeGemmaGreedy,
  readGemmaGreedyResult,
  type GemmaGreedyResult,
} from "./decode-greedy";
import type { GemmaDecodeModelResources } from "./decode-model";
import {
  createGemmaPrefillAttentionResources,
  encodeGemmaPrefillAttention,
  getGemmaPrefillAttentionPipeline,
  updateGemmaPrefillAttention,
  type GemmaPrefillAttentionPipeline,
  type GemmaPrefillAttentionResources,
} from "./prefill-attention";
import {
  createGemmaPrefillAddResources,
  createGemmaPrefillGeluMultiplyResources,
  createGemmaPrefillMultiplyResources,
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
  createGemmaPrefillQatLinearResources,
  encodeGemmaPrefillQatLinear,
  getGemmaPrefillQatLinearPipelines,
  type GemmaPrefillBufferSlice,
  type GemmaPrefillQatLinearPipelines,
  type GemmaPrefillQatLinearResources,
  type GemmaPrefillQatLinearWeights,
} from "./prefill-qat-linear";
import {
  createGemmaPrefillRmsResources,
  encodeGemmaPrefillRms,
  getGemmaPrefillRmsPipeline,
  type GemmaPrefillRmsBufferSlice,
  type GemmaPrefillRmsPipeline,
  type GemmaPrefillRmsResources,
} from "./prefill-rms";
import {
  createGemmaPrefillRopeResources,
  encodeGemmaPrefillRope,
  getGemmaPrefillRopePipeline,
  updateGemmaPrefillRope,
  type GemmaPrefillRopePipeline,
  type GemmaPrefillRopeResources,
} from "./prefill-rope";
import {
  createGemmaPrefillStridedCopyResources,
  encodeGemmaPrefillStridedCopy,
  getGemmaPrefillStridedCopyPipeline,
  updateGemmaPrefillStridedCopy,
  type GemmaPrefillStridedCopyResources,
} from "./prefill-strided-copy";

export const GEMMA_FIXED_PREFILL_ROWS = 32;

const LAYER_COUNT = 35;
const HIDDEN_SIZE = 1536;
const PLE_SIZE = 256;
const QUERY_HEADS = 8;
const HIDDEN_BYTES = HIDDEN_SIZE * 4;
const MAX_HEAD_OUTPUT = 4096;
const MAX_INTERMEDIATE = 12288;
const VOCAB_SIZE = 262144;

interface GemmaPrefillProjection {
  pipelines: GemmaPrefillQatLinearPipelines;
  resources: GemmaPrefillQatLinearResources;
}

interface GemmaPrefillNorm {
  pipeline: GemmaPrefillRmsPipeline;
  resources: GemmaPrefillRmsResources;
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
  postAttentionNorm: GemmaPrefillNorm;
  attentionResidual: GemmaPrefillElementwiseResources;
  preFeedforwardNorm: GemmaPrefillNorm;
  gate: GemmaPrefillProjection;
  up: GemmaPrefillProjection;
  gateActivation: GemmaPrefillElementwiseResources;
  down: GemmaPrefillProjection;
  postFeedforwardNorm: GemmaPrefillNorm;
  feedforwardResidual: GemmaPrefillElementwiseResources;
  pleInputCopy: GemmaPrefillStridedCopyResources;
  pleGate: GemmaPrefillPleProjection;
  pleActivation: GemmaPrefillElementwiseResources;
  pleProjection: GemmaPrefillPleProjection;
  postPleNorm: GemmaPrefillNorm;
  pleResidual: GemmaPrefillElementwiseResources;
  layerScale: GemmaPrefillElementwiseResources;
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
): Promise<GemmaFixedPrefillResources> {
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
          scratch.hiddenNorm,
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
            inputScale: attentionScales.qkvInput,
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
          scratch.hiddenNorm,
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
            inputScale: attentionScales.qkvInput,
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
        ));
        valueCopy = own(createGemmaPrefillStridedCopyResources(
          device,
          stridedCopyPipeline,
          scratch.valueNorm,
          cache.valueBuffer,
          copyParameters(kvFeatures),
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
      const postAttentionNorm = await norm(
        HIDDEN_SIZE,
        true,
        scratch.projection,
        slice(attentionWeights.postAttentionAndPreFeedforwardNorm, 0, HIDDEN_BYTES),
        scratch.residualNorm,
      );
      const attentionResidual = own(createGemmaPrefillAddResources(
        device,
        elementwisePipelines.add,
        input.hidden,
        scratch.residualNorm,
        GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
      ));
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
      const gate = await projection(
        HIDDEN_SIZE,
        intermediateFeatures,
        mlpBits,
        scratch.hiddenNorm,
        {
          packedWeights: mlpWeights.gatePacked,
          rowScales: mlpWeights.gateRowScales,
          inputScale: mlpScales.gateInput,
          outputScale: mlpScales.gateOutput,
        },
        scratch.gate,
      );
      const up = await projection(
        HIDDEN_SIZE,
        intermediateFeatures,
        mlpBits,
        scratch.hiddenNorm,
        {
          packedWeights: mlpWeights.upPacked,
          rowScales: mlpWeights.upRowScales,
          inputScale: mlpScales.gateInput,
          outputScale: mlpScales.upOutput,
        },
        scratch.up,
      );
      const gateActivation = own(createGemmaPrefillGeluMultiplyResources(
        device,
        elementwisePipelines.geluMultiply,
        scratch.gate,
        scratch.up,
        mlpWeights.gateGeluLut,
        scratch.activated,
        GEMMA_FIXED_PREFILL_ROWS * intermediateFeatures,
        mlpScales.gateOutput,
      ));
      const down = await projection(
        intermediateFeatures,
        HIDDEN_SIZE,
        mlpBits,
        scratch.activated,
        {
          packedWeights: mlpWeights.downPacked,
          rowScales: mlpWeights.downRowScales,
          inputScale: mlpScales.downInput,
          outputScale: mlpScales.downOutput,
        },
        scratch.down,
      );
      const postFeedforwardNorm = await norm(
        HIDDEN_SIZE,
        true,
        scratch.down,
        mlpWeights.postFeedforwardNorm,
        scratch.residualNorm,
      );
      const feedforwardResidual = own(createGemmaPrefillAddResources(
        device,
        elementwisePipelines.add,
        input.hidden,
        scratch.residualNorm,
        GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
      ));
      const pleInputCopy = own(createGemmaPrefillStridedCopyResources(
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
      ));
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
      const pleActivation = own(createGemmaPrefillGeluMultiplyResources(
        device,
        elementwisePipelines.geluMultiply,
        scratch.pleGate,
        scratch.pleInput,
        mlpWeights.pleGeluLut,
        scratch.pleActivated,
        GEMMA_FIXED_PREFILL_ROWS * PLE_SIZE,
        mlpScales.pleGateOutput,
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
      const postPleNorm = await norm(
        HIDDEN_SIZE,
        true,
        scratch.pleProjection,
        slice(mlpWeights.postPleNextInputNormAndLayerScale, 0, HIDDEN_BYTES),
        scratch.residualNorm,
      );
      const pleResidual = own(createGemmaPrefillAddResources(
        device,
        elementwisePipelines.add,
        input.hidden,
        scratch.residualNorm,
        GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
      ));
      const layerScale = own(createGemmaPrefillMultiplyResources(
        device,
        elementwisePipelines.multiply,
        input.hidden,
        mlpWeights.postPleNextInputNormAndLayerScale,
        GEMMA_FIXED_PREFILL_ROWS * HIDDEN_SIZE,
        HIDDEN_SIZE * 2,
      ));

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
        preFeedforwardNorm,
        gate,
        up,
        gateActivation,
        down,
        postFeedforwardNorm,
        feedforwardResidual,
        pleInputCopy,
        pleGate,
        pleActivation,
        pleProjection: pleProjectionResult,
        postPleNorm,
        pleResidual,
        layerScale,
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
      )),
    };
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
): void {
  encodeGemmaDecodeInput(encoder, resources.decode.inputPipeline, resources.input);
  for (const layer of resources.layers) {
    encodeNorm(encoder, layer.inputNorm);
    encodeProjection(encoder, layer.query);
    encodeNorm(encoder, layer.queryNorm);
    encodeRope(encoder, layer.queryRope);
    if (layer.key && layer.keyNorm && layer.keyRope && layer.value && layer.valueNorm &&
        layer.keyCopy && layer.valueCopy) {
      encodeProjection(encoder, layer.key);
      encodeNorm(encoder, layer.keyNorm);
      encodeRope(encoder, layer.keyRope);
      encodeProjection(encoder, layer.value);
      encodeNorm(encoder, layer.valueNorm);
      encodeGemmaPrefillStridedCopy(
        encoder,
        resources.stridedCopyPipeline,
        layer.keyCopy,
        GEMMA_FIXED_PREFILL_ROWS,
      );
      encodeGemmaPrefillStridedCopy(
        encoder,
        resources.stridedCopyPipeline,
        layer.valueCopy,
        GEMMA_FIXED_PREFILL_ROWS,
      );
    }
    encodeGemmaPrefillAttention(
      encoder,
      layer.attentionPipeline,
      layer.attention,
      GEMMA_FIXED_PREFILL_ROWS,
    );
    encodeProjection(encoder, layer.outputProjection);
    encodeNorm(encoder, layer.postAttentionNorm);
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.add,
      layer.attentionResidual,
    );
    encodeNorm(encoder, layer.preFeedforwardNorm);
    encodeProjection(encoder, layer.gate);
    encodeProjection(encoder, layer.up);
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.geluMultiply,
      layer.gateActivation,
    );
    encodeProjection(encoder, layer.down);
    encodeNorm(encoder, layer.postFeedforwardNorm);
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.add,
      layer.feedforwardResidual,
    );
    encodeGemmaPrefillStridedCopy(
      encoder,
      resources.stridedCopyPipeline,
      layer.pleInputCopy,
      GEMMA_FIXED_PREFILL_ROWS,
    );
    encodePleProjection(encoder, layer.pleGate);
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.geluMultiply,
      layer.pleActivation,
    );
    encodePleProjection(encoder, layer.pleProjection);
    encodeNorm(encoder, layer.postPleNorm);
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.add,
      layer.pleResidual,
    );
    encodeGemmaPrefillElementwise(
      encoder,
      resources.elementwisePipelines.multiply,
      layer.layerScale,
    );
  }
  encodeNorm(encoder, resources.finalNorm);
  encodeGemmaPrefillStridedCopy(
    encoder,
    resources.stridedCopyPipeline,
    resources.lastRowCopy,
    1,
  );
  encodeProjection(encoder, resources.lmHead);
  encodeGemmaGreedy(
    encoder,
    resources.decode.greedyPipelines,
    resources.decode.greedy,
    true,
  );
}

export async function submitGemmaFixedPrefill(
  device: GPUDevice,
  resources: GemmaFixedPrefillResources,
  position: number,
  validRows: number,
): Promise<GemmaGreedyResult> {
  const encoder = device.createCommandEncoder({ label: "Gemma fixed-32 prefill" });
  encodeGemmaFixedPrefill(encoder, resources);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  for (const cache of resources.decode.stack.ownerCaches.values()) {
    cache.commitWrite(position, validRows);
  }
  return readGemmaGreedyResult(resources.decode.greedy);
}

export function destroyGemmaFixedPrefillResources(
  resources: GemmaFixedPrefillResources,
): void {
  for (const buffer of resources.ownedBuffers.toReversed()) buffer.destroy();
}

function encodeNorm(encoder: GPUCommandEncoder, norm: GemmaPrefillNorm): void {
  encodeGemmaPrefillRms(encoder, norm.pipeline, norm.resources);
}

function encodeProjection(
  encoder: GPUCommandEncoder,
  projection: GemmaPrefillProjection,
): void {
  encodeGemmaPrefillQatLinear(encoder, projection.pipelines, projection.resources);
}

function encodeRope(encoder: GPUCommandEncoder, rope: GemmaPrefillRope): void {
  encodeGemmaPrefillRope(
    encoder,
    rope.pipeline,
    rope.resources,
    GEMMA_FIXED_PREFILL_ROWS,
  );
}

function encodePleProjection(
  encoder: GPUCommandEncoder,
  projection: GemmaPrefillPleProjection,
): void {
  encodeGemmaPrefillPleDense(encoder, projection.pipeline, projection.resources);
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
