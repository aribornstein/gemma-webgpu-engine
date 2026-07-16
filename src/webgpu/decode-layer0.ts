import { loadDecodeAttentionFixture } from "../model/decode-attention-fixture";
import { loadDecodeKNormRopeFixture } from "../model/decode-k-norm-rope-fixture";
import { loadDecodeMlpPleFixture } from "../model/decode-mlp-ple-fixture";
import { loadDecodeOprojNormFixture } from "../model/decode-oproj-norm-fixture";
import { loadDecodeRmsFixture } from "../model/decode-rms-fixture";
import type {
  MaterializedGemmaLayer,
  MaterializedProjection,
} from "../model/gemma-layer-materializer";
import { loadCapturedQatQkvFixture } from "../model/qat-linear-fixture";
import {
  createDecodeAttentionBlockResources,
} from "./decode-attention-block";
import {
  createDecodeMlpPleBlockResources,
} from "./decode-mlp-ple-block";
import {
  encodeGemmaDecodeLayer,
  getGemmaDecodeLayerPipelines,
} from "./decode-layer";
import type { GemmaDecodeLayerPipelines } from "./decode-layer";
import { getWebGpuDevice } from "./device";

const KV_OUT = 256;
const LAYER0_CANONICAL_SOURCE_BYTES = 18_576_466;

export interface DecodeLayer0Result {
  sourceOperators: [
    "com.xenova.gemma4.DecodeRmsSrq",
    "com.xenova.gemma4.DecodeQkvProj",
    "com.xenova.gemma4.DecodeQkNormRope",
    "RMSNorm",
    "Gemma4DecodeAttentionPartial",
    "com.xenova.gemma4.DecodeOprojNorm",
    "com.xenova.gemma4.DecodeGateUpNormPresrq",
    "com.xenova.gemma4.DecodeDownNormAddFused",
    "com.xenova.gemma4.DecodePleGateCodes",
    "com.xenova.gemma4.DecodePleProjNormCodes",
  ];
  implementation: "shared-hidden-pre-mlp-ten-dispatch";
  dispatchesPerToken: 10;
  hiddenMaximumAbsoluteError: number;
  hiddenMaximumRelativeError: number;
  hiddenBitMismatches: number;
  nextInputMaximumAbsoluteError: number;
  nextInputMaximumRelativeError: number;
  nextInputNonzeroBitMismatches: number;
  nextInputSignedZeroBitMismatches: number;
  nextSumMaximumAbsoluteError: number;
  nextSumMaximumRelativeError: number;
  nextSumBitMismatches: number;
  gpuBufferAllocations: number;
  bytesAllocated: number;
  allocationsPerDispatch: 0;
  cpuReadbacksBetweenKernels: 0;
  gpuCopiesBetweenKernels: 0;
}

export async function runDecodeLayer0(): Promise<DecodeLayer0Result> {
  const [rms, qkv, kNorm, attention, oproj, mlp, device] = await Promise.all([
    loadDecodeRmsFixture(),
    loadCapturedQatQkvFixture(),
    loadDecodeKNormRopeFixture(),
    loadDecodeAttentionFixture(),
    loadDecodeOprojNormFixture(),
    loadDecodeMlpPleFixture(),
    getWebGpuDevice(),
  ]);
  if (!device.features.has("subgroups") || !device.features.has("shader-f16")) {
    throw new Error("The complete layer-0 decode plan requires subgroups and shader-f16");
  }

  assertFloat32Exact(rms.expectedOutput, qkv.input, "RMS/SRQ to QKV activation");
  assertFloat32Exact(rms.expectedSum, qkv.inputSum, "RMS/SRQ to QKV sum");
  assertFloat32Exact(qkv.expectedQ, attention.q, "Q projection to attention");
  assertFloat32Exact(qkv.expectedK, kNorm.input, "K projection to K norm");
  assertFloat32Exact(
    kNorm.expectedOutput,
    attention.keyCache.subarray(attention.queryOffset * KV_OUT),
    "K norm to cache",
  );
  assertFloat32Exact(kNorm.cosine, attention.cosine, "K/Q cosine row");
  assertFloat32Exact(kNorm.sine, attention.sine, "K/Q sine row");
  assertFloat32Exact(attention.expectedOutput, oproj.attention, "Attention to O projection");
  assertFloat32Exact(rms.hidden, oproj.hiddenBefore, "Layer residual hidden state");
  assertFloat32Exact(oproj.expectedHidden, mlp.hiddenBeforeDown, "O projection hidden to MLP");
  assertUint16Exact(oproj.expectedFfnInputBits, mlp.preMlpInputBits, "O projection input to gate/up");
  assertFloat32Exact(oproj.expectedFfnInputSum, mlp.preMlpSum, "O projection sum to gate/up");
  const materialized = materializedLayer0FromFixtures(rms, qkv, kNorm, attention, oproj, mlp);

  const pipelines: GemmaDecodeLayerPipelines = await getGemmaDecodeLayerPipelines(
    device,
    materialized.profile,
  );

  const attentionResources = createDecodeAttentionBlockResources(
    device,
    pipelines.attention,
    rms,
    qkv,
    kNorm,
    attention,
    oproj,
    materialized,
  );
  const mlpResources = createDecodeMlpPleBlockResources(device, pipelines.mlp, mlp, {
    preMlpInput: attentionResources.ffnInputBuffer,
    preMlpSum: attentionResources.ffnInputSumBuffer,
    hidden: attentionResources.hiddenBuffer,
  }, {
    layer: materialized,
    pleNormWeights: mlp.pleNormWeights,
    nextInputScale: 0.4842597544193268,
  });
  const hiddenBytes = mlp.expectedHiddenAfterPle.byteLength;
  const nextInputBytes = mlp.expectedNextLayerInput.byteLength;
  const readback = device.createBuffer({
    label: "Complete layer-0 readback",
    size: hiddenBytes + nextInputBytes + 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    device.queue.writeBuffer(attentionResources.hiddenBuffer, 0, rms.hidden);
    const encoder = device.createCommandEncoder({ label: "Complete layer-0 decode" });
    encodeGemmaDecodeLayer(encoder, pipelines, {
      attention: attentionResources,
      mlp: mlpResources,
    });
    encoder.copyBufferToBuffer(mlpResources.hidden, 0, readback, 0, hiddenBytes);
    encoder.copyBufferToBuffer(
      mlpResources.nextInput,
      0,
      readback,
      hiddenBytes,
      nextInputBytes,
    );
    encoder.copyBufferToBuffer(
      mlpResources.nextSum,
      0,
      readback,
      hiddenBytes + nextInputBytes,
      4,
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const actualHidden = new Float32Array(mapped.slice(0, hiddenBytes));
    const actualNextInput = new Float32Array(
      mapped.slice(hiddenBytes, hiddenBytes + nextInputBytes),
    );
    const actualNextSum = new Float32Array(mapped.slice(hiddenBytes + nextInputBytes));
    readback.unmap();
    attentionResources.cache.commitWrite(attentionResources.cachePosition);

    const hiddenErrors = measureErrors(actualHidden, mlp.expectedHiddenAfterPle);
    const nextInputErrors = measureErrors(actualNextInput, mlp.expectedNextLayerInput);
    const nextSumErrors = measureErrors(actualNextSum, mlp.expectedNextLayerSum);
    const buffers = [...attentionResources.buffers, ...mlpResources.buffers, readback];
    return {
      sourceOperators: [
        "com.xenova.gemma4.DecodeRmsSrq",
        "com.xenova.gemma4.DecodeQkvProj",
        "com.xenova.gemma4.DecodeQkNormRope",
        "RMSNorm",
        "Gemma4DecodeAttentionPartial",
        "com.xenova.gemma4.DecodeOprojNorm",
        "com.xenova.gemma4.DecodeGateUpNormPresrq",
        "com.xenova.gemma4.DecodeDownNormAddFused",
        "com.xenova.gemma4.DecodePleGateCodes",
        "com.xenova.gemma4.DecodePleProjNormCodes",
      ],
      implementation: "shared-hidden-pre-mlp-ten-dispatch",
      dispatchesPerToken: 10,
      hiddenMaximumAbsoluteError: hiddenErrors.maximumAbsoluteError,
      hiddenMaximumRelativeError: hiddenErrors.maximumRelativeError,
      hiddenBitMismatches: hiddenErrors.bitMismatches,
      nextInputMaximumAbsoluteError: nextInputErrors.maximumAbsoluteError,
      nextInputMaximumRelativeError: nextInputErrors.maximumRelativeError,
      nextInputNonzeroBitMismatches: nextInputErrors.nonzeroBitMismatches,
      nextInputSignedZeroBitMismatches: nextInputErrors.signedZeroBitMismatches,
      nextSumMaximumAbsoluteError: nextSumErrors.maximumAbsoluteError,
      nextSumMaximumRelativeError: nextSumErrors.maximumRelativeError,
      nextSumBitMismatches: nextSumErrors.bitMismatches,
      gpuBufferAllocations: buffers.length + attentionResources.cache.buffers.length,
      bytesAllocated:
        buffers.reduce((sum, buffer) => sum + buffer.size, 0) +
        attentionResources.cache.bytesAllocated,
      allocationsPerDispatch: 0,
      cpuReadbacksBetweenKernels: 0,
      gpuCopiesBetweenKernels: 0,
    };
  } finally {
    readback.destroy();
    for (const buffer of mlpResources.buffers) buffer.destroy();
    for (const buffer of attentionResources.buffers) buffer.destroy();
    attentionResources.cache.destroy();
  }
}

function materializedLayer0FromFixtures(
  rms: Awaited<ReturnType<typeof loadDecodeRmsFixture>>,
  qkv: Awaited<ReturnType<typeof loadCapturedQatQkvFixture>>,
  kNorm: Awaited<ReturnType<typeof loadDecodeKNormRopeFixture>>,
  attention: Awaited<ReturnType<typeof loadDecodeAttentionFixture>>,
  oproj: Awaited<ReturnType<typeof loadDecodeOprojNormFixture>>,
  mlp: Awaited<ReturnType<typeof loadDecodeMlpPleFixture>>,
): MaterializedGemmaLayer {
  const projection = (
    packedWeights: Uint32Array,
    rowScales: Float32Array,
    inputScale: number,
    outputScale: number,
  ): MaterializedProjection => ({ packedWeights, rowScales, inputScale, outputScale });
  return {
    layerIndex: 0,
    profile: "sliding-int4",
    qkv: {
      packedWeights: qkv.packedWeights,
      rowScales: qkv.rowScales,
      inputScale: rms.inputScale,
      outputScales: qkv.outputScales,
    },
    outputProjection: projection(
      oproj.packedWeights,
      oproj.rowScales,
      attention.outputQuantScale,
      oproj.outputScale,
    ),
    mlp: {
      gate: projection(
        mlp.gateBits,
        mlp.gateScales,
        oproj.inScale2,
        0.6181102395057678,
      ),
      up: projection(
        mlp.upBits,
        mlp.upScales,
        oproj.inScale2,
        0.6181102395057678,
      ),
      down: projection(
        mlp.downBits,
        mlp.downScales,
        27.842519760131836,
        16.64207649230957,
      ),
    },
    ple: {
      inputGate: projection(
        mlp.pleGateWeights,
        mlp.pleGateRowScales,
        3.334678888320923,
        0.01857776567339897,
      ),
      projection: projection(
        mlp.pleProjectionWeights,
        mlp.pleProjectionRowScales,
        0.03764764964580536,
        0.03129800781607628,
      ),
    },
    norms: {
      input: rms.weight,
      q: attention.qNormWeight,
      k: kNorm.weight,
      postAttention: oproj.normWeights.slice(0, 1536),
      preFeedforward: oproj.normWeights.slice(1536),
      postFeedforward: mlp.postFfNorm,
      postPerLayerInput: mlp.pleNormWeights.slice(0, 1536),
      oProjectionFused: oproj.normWeights,
    },
    layerScalar: mlp.pleNormWeights[3072],
    sourceBytes: LAYER0_CANONICAL_SOURCE_BYTES,
  };
}

function assertFloat32Exact(actual: Float32Array, expected: Float32Array, boundary: string): void {
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  if (actualBits.length !== expectedBits.length) throw new Error(`${boundary} length mismatch`);
  for (let index = 0; index < actualBits.length; index += 1) {
    if (actualBits[index] !== expectedBits[index]) throw new Error(`${boundary} differs at ${index}`);
  }
}

function assertUint16Exact(actual: Uint16Array, expected: Uint16Array, boundary: string): void {
  if (actual.length !== expected.length) throw new Error(`${boundary} length mismatch`);
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) throw new Error(`${boundary} differs at ${index}`);
  }
}

function measureErrors(actual: Float32Array, expected: Float32Array) {
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  let bitMismatches = 0;
  let signedZeroBitMismatches = 0;
  let nonzeroBitMismatches = 0;
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const absolute = Math.abs(actual[index] - expected[index]);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absolute);
    maximumRelativeError = Math.max(
      maximumRelativeError,
      absolute / Math.max(Math.abs(expected[index]), 1e-7),
    );
    if (actualBits[index] !== expectedBits[index]) {
      bitMismatches += 1;
      if (
        (actualBits[index] & 0x7fffffff) === 0 &&
        (expectedBits[index] & 0x7fffffff) === 0
      ) {
        signedZeroBitMismatches += 1;
      } else {
        nonzeroBitMismatches += 1;
      }
    }
  }
  return {
    maximumAbsoluteError,
    maximumRelativeError,
    bitMismatches,
    signedZeroBitMismatches,
    nonzeroBitMismatches,
  };
}