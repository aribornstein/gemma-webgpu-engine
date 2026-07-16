import type { CachedTensorPayload } from "./cached-safetensors.ts";
import type { GemmaProjectionPlan } from "./gemma-layer-plan.ts";
import type { LoadedGemmaLayerWeights } from "./gemma-layer-weights.ts";

const HIDDEN_SIZE = 1536;

export interface MaterializedProjection {
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  inputScale: number;
  outputScale: number;
}

export interface MaterializedGemmaLayer {
  layerIndex: number;
  profile: LoadedGemmaLayerWeights["plan"]["profile"];
  qkv: {
    packedWeights: Uint32Array;
    rowScales: Float32Array;
    inputScale: number;
    outputScales: Float32Array;
  };
  outputProjection: MaterializedProjection;
  mlp: {
    gate: MaterializedProjection;
    up: MaterializedProjection;
    down: MaterializedProjection;
  };
  ple: {
    inputGate: MaterializedProjection;
    projection: MaterializedProjection;
  };
  norms: {
    input: Float32Array;
    q: Float32Array;
    k: Float32Array | null;
    postAttention: Float32Array;
    preFeedforward: Float32Array;
    postFeedforward: Float32Array;
    postPerLayerInput: Float32Array;
    oProjectionFused: Float32Array;
  };
  layerScalar: number;
  sourceBytes: number;
}

export function materializeGemmaLayerWeights(
  weights: LoadedGemmaLayerWeights,
): MaterializedGemmaLayer {
  const { plan } = weights;
  const q = materializeProjection(weights, plan.attention.q);
  const k = plan.attention.k ? materializeProjection(weights, plan.attention.k) : null;
  const v = plan.attention.v ? materializeProjection(weights, plan.attention.v) : null;
  if ((k && !sameFloatBits(q.inputScale, k.inputScale)) ||
      (v && !sameFloatBits(q.inputScale, v.inputScale))) {
    throw new Error(`Gemma layer ${plan.layerIndex} Q/K/V input scales do not match`);
  }

  const input = bfloat16ToFloat32(tensor(weights, plan.norms.input.name));
  const postAttention = bfloat16ToFloat32(tensor(weights, plan.norms.postAttention.name));
  const preFeedforward = bfloat16ToFloat32(tensor(weights, plan.norms.preFeedforward.name));
  const postFeedforward = bfloat16ToFloat32(tensor(weights, plan.norms.postFeedforward.name));
  const postPerLayerInput = bfloat16ToFloat32(
    tensor(weights, plan.norms.postPerLayerInput.name),
  );

  return {
    layerIndex: plan.layerIndex,
    profile: plan.profile,
    qkv: {
      packedWeights: concatenateUint32([
        q.packedWeights,
        ...(k ? [k.packedWeights] : []),
        ...(v ? [v.packedWeights] : []),
      ]),
      rowScales: concatenateFloat32([
        q.rowScales,
        ...(k ? [k.rowScales] : []),
        ...(v ? [v.rowScales] : []),
      ]),
      inputScale: q.inputScale,
      outputScales: new Float32Array([
        q.outputScale,
        k?.outputScale ?? 0,
        v?.outputScale ?? 0,
      ]),
    },
    outputProjection: materializeProjection(weights, plan.attention.output),
    mlp: {
      gate: materializeProjection(weights, plan.mlp.gate),
      up: materializeProjection(weights, plan.mlp.up),
      down: materializeProjection(weights, plan.mlp.down),
    },
    ple: {
      inputGate: materializeProjection(weights, plan.ple.inputGate),
      projection: materializeProjection(weights, plan.ple.projection),
    },
    norms: {
      input,
      q: bfloat16ToFloat32(tensor(weights, plan.attention.qNorm.name)),
      k: plan.attention.kNorm
        ? bfloat16ToFloat32(tensor(weights, plan.attention.kNorm.name))
        : null,
      postAttention,
      preFeedforward,
      postFeedforward,
      postPerLayerInput,
      oProjectionFused: concatenateFloat32([postAttention, preFeedforward]),
    },
    layerScalar: float32ScalarFromBfloat16(tensor(weights, plan.layerScalar.name)),
    sourceBytes: weights.bytesLoaded,
  };
}

export function createPleNormWeights(
  layer: MaterializedGemmaLayer,
  nextInputNorm: Float32Array,
): Float32Array {
  if (nextInputNorm.length !== HIDDEN_SIZE) {
    throw new Error(`Next input norm must contain ${HIDDEN_SIZE} values`);
  }
  return concatenateFloat32([
    layer.norms.postPerLayerInput,
    nextInputNorm,
    new Float32Array([layer.layerScalar]),
  ]);
}

export function materializeProjection(
  weights: LoadedGemmaLayerWeights,
  projection: GemmaProjectionPlan,
): MaterializedProjection {
  return {
    packedWeights: packedUint32(tensor(weights, projection.weight.name)),
    rowScales: float32LittleEndian(tensor(weights, projection.weightScale.name)),
    inputScale: float32Scalar(tensor(weights, projection.inputActivationScale.name)),
    outputScale: float32Scalar(tensor(weights, projection.outputActivationScale.name)),
  };
}

export function packedUint32(tensorPayload: CachedTensorPayload): Uint32Array {
  if ((tensorPayload.dtype !== "U8" && tensorPayload.dtype !== "I8") ||
      tensorPayload.bytes.byteLength % 4 !== 0) {
    throw new Error(`Tensor ${tensorPayload.name} is not packed byte storage aligned to u32`);
  }
  const bytes = tensorPayload.dtype === "I8"
    ? Uint8Array.from(tensorPayload.bytes, (value) => (value + 128) & 0xff)
    : tensorPayload.bytes;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint32Array.from(
    { length: tensorPayload.bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  );
}

export function float32LittleEndian(tensorPayload: CachedTensorPayload): Float32Array {
  if (tensorPayload.dtype !== "F32" || tensorPayload.bytes.byteLength % 4 !== 0) {
    throw new Error(`Tensor ${tensorPayload.name} is not F32 storage`);
  }
  const view = new DataView(
    tensorPayload.bytes.buffer,
    tensorPayload.bytes.byteOffset,
    tensorPayload.bytes.byteLength,
  );
  return Float32Array.from(
    { length: tensorPayload.bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}

export function bfloat16ToFloat32(tensorPayload: CachedTensorPayload): Float32Array {
  if (tensorPayload.dtype !== "BF16" || tensorPayload.bytes.byteLength % 2 !== 0) {
    throw new Error(`Tensor ${tensorPayload.name} is not BF16 storage`);
  }
  const source = new DataView(
    tensorPayload.bytes.buffer,
    tensorPayload.bytes.byteOffset,
    tensorPayload.bytes.byteLength,
  );
  const bits = new Uint32Array(tensorPayload.bytes.byteLength / 2);
  for (let index = 0; index < bits.length; index += 1) {
    bits[index] = source.getUint16(index * 2, true) << 16;
  }
  return new Float32Array(bits.buffer);
}

function float32Scalar(tensorPayload: CachedTensorPayload): number {
  const values = float32LittleEndian(tensorPayload);
  if (values.length !== 1) throw new Error(`Tensor ${tensorPayload.name} is not an F32 scalar`);
  return values[0];
}

function float32ScalarFromBfloat16(tensorPayload: CachedTensorPayload): number {
  const values = bfloat16ToFloat32(tensorPayload);
  if (values.length !== 1) throw new Error(`Tensor ${tensorPayload.name} is not a BF16 scalar`);
  return values[0];
}

function tensor(
  weights: LoadedGemmaLayerWeights,
  name: string,
): CachedTensorPayload {
  const tensorPayload = weights.tensors.get(name);
  if (!tensorPayload) throw new Error(`Loaded Gemma layer is missing tensor ${name}`);
  return tensorPayload;
}

function concatenateFloat32(arrays: readonly Float32Array[]): Float32Array {
  const output = new Float32Array(arrays.reduce((total, array) => total + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    output.set(array, offset);
    offset += array.length;
  }
  return output;
}

function concatenateUint32(arrays: readonly Uint32Array[]): Uint32Array {
  const output = new Uint32Array(arrays.reduce((total, array) => total + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    output.set(array, offset);
    offset += array.length;
  }
  return output;
}

function sameFloatBits(left: number, right: number): boolean {
  const values = new Float32Array([left, right]);
  const bits = new Uint32Array(values.buffer);
  return bits[0] === bits[1];
}