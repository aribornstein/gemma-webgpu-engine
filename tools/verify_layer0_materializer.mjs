import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import {
  bfloat16ToFloat32,
  createPleNormWeights,
  materializeGemmaLayerWeights,
} from "../src/model/gemma-layer-materializer.ts";
import { createGemmaLayerPlan } from "../src/model/gemma-layer-plan.ts";

const REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";
const MODEL_URL =
  `https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/${REVISION}/model.safetensors`;
const HEADER_BYTES = 375400;
const SOURCE_SIZE = 2458111846;

const headerContainer = await fetchRange(0, HEADER_BYTES);
const headerLength = Number(new DataView(
  headerContainer.buffer,
  headerContainer.byteOffset,
  8,
).getBigUint64(0, true));
if (headerLength !== 375392) throw new Error(`Unexpected header length ${headerLength}`);
const dataStart = 8 + headerLength;
const header = JSON.parse(
  new TextDecoder().decode(headerContainer.subarray(8, dataStart)).trim(),
);
if (Object.keys(header).filter((name) => name !== "__metadata__").length !== 2780) {
  throw new Error("Pinned safetensors tensor count changed");
}

const layerPrefix = "model.language_model.layers.0.";
const layerEntries = Object.entries(header).filter(([name]) => name.startsWith(layerPrefix));
const nextNormName = "model.language_model.layers.1.input_layernorm.weight";
const requested = [...layerEntries, [nextNormName, header[nextNormName]]];
const payloads = await Promise.all(requested.map(async ([name, descriptor]) => {
  const begin = dataStart + descriptor.data_offsets[0];
  const end = dataStart + descriptor.data_offsets[1];
  const bytes = await fetchRange(begin, end);
  return [name, {
    name,
    dtype: descriptor.dtype,
    shape: descriptor.shape,
    begin,
    end,
    byteLength: end - begin,
    bytes,
    sha256: sha256(bytes),
  }];
}));
const tensors = new Map(payloads);
const layerDescriptors = new Map(layerEntries.map(([name]) => [name, tensors.get(name)]));
const plan = createGemmaLayerPlan(layerDescriptors, 0);
const materialized = materializeGemmaLayerWeights({
  plan,
  tensors: layerDescriptors,
  tensorHashes: new Map(layerEntries.map(([name]) => [name, tensors.get(name).sha256])),
  bytesLoaded: plan.tensorBytes,
});
const nextInputNorm = bfloat16ToFloat32(tensors.get(nextNormName));
const pleNormWeights = createPleNormWeights(materialized, nextInputNorm);

const [qkv, rms, kNorm, attention, oproj, mlp] = await Promise.all([
  readSafetensors("public/models/gemma-4-e2b/operators/layer0-qkv.safetensors"),
  readSafetensors("public/models/gemma-4-e2b/operators/decode-rms-srq-layer0.safetensors"),
  readSafetensors("public/models/gemma-4-e2b/operators/decode-k-norm-rope-layer0.safetensors"),
  readSafetensors("public/models/gemma-4-e2b/operators/decode-attention-layer0.safetensors"),
  readSafetensors("public/models/gemma-4-e2b/operators/decode-oproj-norm-layer0.safetensors"),
  readSafetensors("public/models/gemma-4-e2b/operators/decode-mlp-ple-layer0.safetensors"),
]);

const qPrefix = `${layerPrefix}self_attn.q_proj`;
const kPrefix = `${layerPrefix}self_attn.k_proj`;
const vPrefix = `${layerPrefix}self_attn.v_proj`;
expectBytes(materialized.qkv.packedWeights, concatenate([
  qkv.get(`${qPrefix}.weight`),
  qkv.get(`${kPrefix}.weight`),
  qkv.get(`${vPrefix}.weight`),
]), "QKV packed weights");
expectBytes(materialized.qkv.rowScales, concatenate([
  qkv.get(`${qPrefix}.weight_scale`),
  qkv.get(`${kPrefix}.weight_scale`),
  qkv.get(`${vPrefix}.weight_scale`),
]), "QKV row scales");
expectBytes(materialized.norms.input, rms.get("weight"), "input RMS norm");
expectBytes(materialized.norms.k, kNorm.get("weight"), "K RMS norm");
expectBytes(materialized.norms.q, attention.get("q_norm_weight"), "Q RMS norm");
expectBytes(materialized.outputProjection.packedWeights, oproj.get("packed_weights"), "O packed weights");
expectBytes(materialized.outputProjection.rowScales, oproj.get("row_scales"), "O row scales");
expectBytes(materialized.norms.oProjectionFused, oproj.get("norm_weights"), "O fused norms");
expectBytes(materialized.mlp.gate.packedWeights, mlp.get("gateBits"), "gate packed weights");
expectBytes(materialized.mlp.gate.rowScales, mlp.get("gateScales"), "gate row scales");
expectBytes(materialized.mlp.up.packedWeights, mlp.get("upBits"), "up packed weights");
expectBytes(materialized.mlp.up.rowScales, mlp.get("upScales"), "up row scales");
expectBytes(materialized.mlp.down.packedWeights, mlp.get("downBits"), "down packed weights");
expectBytes(materialized.mlp.down.rowScales, mlp.get("downScales"), "down row scales");
expectBytes(materialized.norms.postFeedforward, mlp.get("postFfNorm"), "post-FFN norm");
expectBytes(materialized.ple.inputGate.packedWeights, mlp.get("pleGateWeights"), "PLE gate weights");
expectBytes(materialized.ple.inputGate.rowScales, mlp.get("pleGateRowScales"), "PLE gate scales");
expectBytes(
  materialized.ple.projection.packedWeights,
  mlp.get("pleProjectionWeights"),
  "PLE projection weights",
);
expectBytes(
  materialized.ple.projection.rowScales,
  mlp.get("pleProjectionRowScales"),
  "PLE projection scales",
);
expectBytes(pleNormWeights, mlp.get("pleNormWeights"), "PLE fused norms");

console.log(JSON.stringify({
  revision: REVISION,
  layer: 0,
  profile: materialized.profile,
  sourceBytes: materialized.sourceBytes,
  comparisons: 19,
  exact: true,
}));

async function fetchRange(begin, end) {
  const response = await fetch(MODEL_URL, {
    headers: { Range: `bytes=${begin}-${end - 1}` },
  });
  if (response.status !== 206) throw new Error(`Range ${begin}-${end} returned ${response.status}`);
  const contentRange = response.headers.get("content-range");
  if (contentRange !== `bytes ${begin}-${end - 1}/${SOURCE_SIZE}`) {
    throw new Error(`Range identity mismatch: ${contentRange}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== end - begin) throw new Error(`Range ${begin}-${end} is truncated`);
  return bytes;
}

async function readSafetensors(path) {
  const bytes = new Uint8Array(await readFile(path));
  const headerLength = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerLength;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, dataStart)).trim());
  return new Map(Object.entries(header)
    .filter(([name]) => name !== "__metadata__")
    .map(([name, descriptor]) => [
      name,
      bytes.slice(dataStart + descriptor.data_offsets[0], dataStart + descriptor.data_offsets[1]),
    ]));
}

function expectBytes(actual, expected, label) {
  if (!actual) throw new Error(`${label} materialization is missing`);
  const actualBytes = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength);
  if (actualBytes.byteLength !== expected.byteLength ||
      actualBytes.some((value, index) => value !== expected[index])) {
    throw new Error(`${label} does not match the exact fixture`);
  }
}

function concatenate(arrays) {
  const bytes = new Uint8Array(arrays.reduce((total, array) => total + array.byteLength, 0));
  let offset = 0;
  for (const array of arrays) {
    bytes.set(array, offset);
    offset += array.byteLength;
  }
  return bytes;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}