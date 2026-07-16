import { modelAssetUrl } from "./model-assets";

const FIXTURE_URL = modelAssetUrl(
  "gemma-4-e2b/operators/decode-oproj-norm-layer0.safetensors",
);
const CONTAINER_SHA256 = "d8ec21da0edcccdfd478c76e90215b79d4bae5a4f58eebf8b1de355c474a223d";
const SOURCE_METADATA_SHA256 = "1b81dd537bc0418ce74d93ee5dcf8c0b5d4b70c4c8621bb2ec3ae15c3d0dacdf";
const SOURCE_TENSORS_SHA256 = "b5eff21d1af5f8826cd00a2a01d0830462fad78505a803c50ef1ca12b8e2ac52";

const EXPECTED = {
  attention: { dtype: "F32", shape: [1, 2048], sha256: "3273707bd3456f41e904caa6aeaad622db0abda6c972840c3d8d69310bd913eb" },
  packed_weights: { dtype: "U32", shape: [1536, 256], sha256: "a6bccc46f469f441df077de2d07b4b07cc3b1b14358384938a139562eb76f6e2" },
  row_scales: { dtype: "F32", shape: [1536], sha256: "f2070a12a9ecf0c3a8874cb19c345253461edb517cae7f062c795cbee7b75e8e" },
  hidden_before: { dtype: "F32", shape: [1, 1536], sha256: "4b7c92982f60cc985f150bdbb22a28a8e276bf516edd2b6a24c284fb810ed6a2" },
  norm_weights: { dtype: "F32", shape: [2, 1536], sha256: "0ed24ca107f0365f5e72538f1042c566b7c9cf73fb418c5183c8aa510ae122d8" },
  hidden_after: { dtype: "F32", shape: [1, 1536], sha256: "6bfbb8d43ec4bee447817403dd693194d438b3a3b9b390eca438922d77ac7e6a" },
  ffn_input: { dtype: "F16", shape: [1, 1536], sha256: "b5be5add660db27528dc03e0ce096b435db77f97a2429aeadc56252ba637e487" },
  ffn_input_sum: { dtype: "F32", shape: [1], sha256: "7e7adb267c616e101e6817e8426bc641289047bc2624e955abb90aeb25ee700f" },
} as const;

interface TensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface DecodeOprojNormFixture {
  attention: Float32Array;
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  hiddenBefore: Float32Array;
  normWeights: Float32Array;
  expectedHidden: Float32Array;
  expectedFfnInputBits: Uint16Array;
  expectedFfnInputSum: Float32Array;
  bits: 4;
  inFeatures: 2048;
  outFeatures: 1536;
  outputScale: 0.21056734025478363;
  epsilon: 0.000001;
  inScale2: 0.9406865835189819;
  artifactSha256: string;
  sourceMetadataSha256: string;
  sourceTensorsSha256: string;
}

let fixturePromise: Promise<DecodeOprojNormFixture> | null = null;

export function loadDecodeOprojNormFixture(): Promise<DecodeOprojNormFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

async function loadFixture(): Promise<DecodeOprojNormFixture> {
  const response = await fetch(FIXTURE_URL);
  if (!response.ok) throw new Error(`Failed to load DecodeOprojNorm fixture: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const artifactSha256 = await sha256(bytes);
  if (artifactSha256 !== CONTAINER_SHA256) {
    throw new Error(`DecodeOprojNorm fixture hash mismatch: ${artifactSha256}`);
  }
  const tensors = await readExpectedTensors(bytes);
  return {
    attention: float32LittleEndian(requiredTensor(tensors, "attention")),
    packedWeights: uint32LittleEndian(requiredTensor(tensors, "packed_weights")),
    rowScales: float32LittleEndian(requiredTensor(tensors, "row_scales")),
    hiddenBefore: float32LittleEndian(requiredTensor(tensors, "hidden_before")),
    normWeights: float32LittleEndian(requiredTensor(tensors, "norm_weights")),
    expectedHidden: float32LittleEndian(requiredTensor(tensors, "hidden_after")),
    expectedFfnInputBits: uint16LittleEndian(requiredTensor(tensors, "ffn_input")),
    expectedFfnInputSum: float32LittleEndian(requiredTensor(tensors, "ffn_input_sum")),
    bits: 4,
    inFeatures: 2048,
    outFeatures: 1536,
    outputScale: 0.21056734025478363,
    epsilon: 0.000001,
    inScale2: 0.9406865835189819,
    artifactSha256,
    sourceMetadataSha256: SOURCE_METADATA_SHA256,
    sourceTensorsSha256: SOURCE_TENSORS_SHA256,
  };
}

async function readExpectedTensors(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  if (bytes.byteLength < 8) throw new Error("DecodeOprojNorm fixture is missing its prefix");
  const headerBytes = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerBytes;
  if (!Number.isSafeInteger(headerBytes) || dataStart > bytes.byteLength) {
    throw new Error("DecodeOprojNorm fixture has an invalid header length");
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, dataStart)).trim(),
  ) as Record<string, unknown>;
  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(EXPECTED)) {
    const tensor = parseTensorHeader(header[name], name);
    if (tensor.dtype !== expected.dtype || tensor.shape.join(",") !== expected.shape.join(",")) {
      throw new Error(`DecodeOprojNorm fixture metadata mismatch for ${name}`);
    }
    const payload = bytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const actualSha256 = await sha256(payload);
    if (actualSha256 !== expected.sha256) {
      throw new Error(`DecodeOprojNorm tensor hash mismatch for ${name}: ${actualSha256}`);
    }
    tensors.set(name, payload);
  }
  return tensors;
}

function parseTensorHeader(value: unknown, name: string): TensorHeader {
  if (typeof value !== "object" || value === null) {
    throw new Error(`DecodeOprojNorm fixture is missing tensor ${name}`);
  }
  const tensor = value as Partial<TensorHeader>;
  if (
    typeof tensor.dtype !== "string" || !Array.isArray(tensor.shape) ||
    !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2
  ) {
    throw new Error(`DecodeOprojNorm fixture has an invalid header for ${name}`);
  }
  return tensor as TensorHeader;
}

function requiredTensor(tensors: Map<string, Uint8Array>, name: string): Uint8Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`DecodeOprojNorm fixture is missing payload ${name}`);
  return tensor;
}

function float32LittleEndian(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true));
}

function uint32LittleEndian(bytes: Uint8Array): Uint32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint32Array.from({ length: bytes.byteLength / 4 }, (_, index) => view.getUint32(index * 4, true));
}

function uint16LittleEndian(bytes: Uint8Array): Uint16Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint16Array.from({ length: bytes.byteLength / 2 }, (_, index) => view.getUint16(index * 2, true));
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}