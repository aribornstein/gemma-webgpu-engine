import { modelAssetUrl } from "./model-assets";

const FIXTURE_URL = modelAssetUrl(
  "gemma-4-e2b/operators/decode-attention-layer0.safetensors",
);
const CONTAINER_SHA256 = "4ad8f65ebbaf1f71fbcb4ea20e22906e5bd4fa2765b077a6ddf3183b22277b97";
const SOURCE_CAPTURE_SHA256 = "fa4d670c13f3f7e1d271040b994aefb85106fe0b0343646907fb54cc9b907f2b";

const EXPECTED = {
  q: {
    dtype: "F32",
    shape: [1, 8, 256],
    sha256: "ab887bc85137320455da9e3a9b5b8e121bb19dced68326cfa1434f82b951482b",
  },
  q_norm_weight: {
    dtype: "F32",
    shape: [256],
    sha256: "9bd79dd6adec377becdb3b6438691fa38b20e17c40826eb5698e16086093b347",
  },
  cosine: {
    dtype: "F32",
    shape: [1, 128],
    sha256: "0a8d34ce643de6612a8725b36cb6599f3de2bb956fbd3c96f7577014c9707141",
  },
  sine: {
    dtype: "F32",
    shape: [1, 128],
    sha256: "078b5542f3148c0eeb381bd8be0bec9b2770f366f0466093c984e659807b5132",
  },
  key_cache: {
    dtype: "F32",
    shape: [11, 1, 256],
    sha256: "7bbcbafa49c38c6897649de552926c05d4737120dda4c81bbb4b3f4a1d242fd8",
  },
  value_cache: {
    dtype: "F32",
    shape: [11, 1, 256],
    sha256: "87f0773acac7e9b6ab2814d3fd4771cf6c803223546379bf521f097e21fe0f56",
  },
  output: {
    dtype: "F32",
    shape: [1, 8, 256],
    sha256: "3273707bd3456f41e904caa6aeaad622db0abda6c972840c3d8d69310bd913eb",
  },
} as const;

interface TensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface DecodeAttentionFixture {
  q: Float32Array;
  qNormWeight: Float32Array;
  cosine: Float32Array;
  sine: Float32Array;
  keyCache: Float32Array;
  valueCache: Float32Array;
  expectedOutput: Float32Array;
  qHeads: 8;
  kvHeads: 1;
  headDim: 256;
  keyLength: 11;
  queryOffset: 10;
  window: 512;
  epsilon: 0.000001;
  outputQuantScale: 0.03026575781404972;
  artifactUrl: string;
  artifactSha256: string;
  sourceCaptureSha256: string;
}

let fixturePromise: Promise<DecodeAttentionFixture> | null = null;

export function loadDecodeAttentionFixture(): Promise<DecodeAttentionFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

async function loadFixture(): Promise<DecodeAttentionFixture> {
  const response = await fetch(FIXTURE_URL);
  if (!response.ok) {
    throw new Error(`Failed to load decode attention fixture: ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const artifactSha256 = await sha256(bytes);
  if (artifactSha256 !== CONTAINER_SHA256) {
    throw new Error(`Decode attention fixture hash mismatch: ${artifactSha256}`);
  }
  const tensors = await readExpectedTensors(bytes);
  return {
    q: float32LittleEndian(requiredTensor(tensors, "q")),
    qNormWeight: float32LittleEndian(requiredTensor(tensors, "q_norm_weight")),
    cosine: float32LittleEndian(requiredTensor(tensors, "cosine")),
    sine: float32LittleEndian(requiredTensor(tensors, "sine")),
    keyCache: float32LittleEndian(requiredTensor(tensors, "key_cache")),
    valueCache: float32LittleEndian(requiredTensor(tensors, "value_cache")),
    expectedOutput: float32LittleEndian(requiredTensor(tensors, "output")),
    qHeads: 8,
    kvHeads: 1,
    headDim: 256,
    keyLength: 11,
    queryOffset: 10,
    window: 512,
    epsilon: 0.000001,
    outputQuantScale: 0.03026575781404972,
    artifactUrl: FIXTURE_URL,
    artifactSha256,
    sourceCaptureSha256: SOURCE_CAPTURE_SHA256,
  };
}

async function readExpectedTensors(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  if (bytes.byteLength < 8) throw new Error("Decode attention fixture is missing its prefix");
  const headerBytes = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerBytes;
  if (!Number.isSafeInteger(headerBytes) || dataStart > bytes.byteLength) {
    throw new Error("Decode attention fixture has an invalid header length");
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, dataStart)).trim(),
  ) as Record<string, unknown>;
  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(EXPECTED)) {
    const tensor = parseTensorHeader(header[name], name);
    if (tensor.dtype !== expected.dtype || tensor.shape.join(",") !== expected.shape.join(",")) {
      throw new Error(`Decode attention fixture metadata mismatch for ${name}`);
    }
    const payload = bytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const actualSha256 = await sha256(payload);
    if (actualSha256 !== expected.sha256) {
      throw new Error(`Decode attention tensor hash mismatch for ${name}: ${actualSha256}`);
    }
    tensors.set(name, payload);
  }
  return tensors;
}

function parseTensorHeader(value: unknown, name: string): TensorHeader {
  if (typeof value !== "object" || value === null) {
    throw new Error(`Decode attention fixture is missing tensor ${name}`);
  }
  const tensor = value as Partial<TensorHeader>;
  if (
    typeof tensor.dtype !== "string" || !Array.isArray(tensor.shape) ||
    !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2
  ) {
    throw new Error(`Decode attention fixture has an invalid header for ${name}`);
  }
  return tensor as TensorHeader;
}

function requiredTensor(tensors: Map<string, Uint8Array>, name: string): Uint8Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`Decode attention fixture is missing payload ${name}`);
  return tensor;
}

function float32LittleEndian(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("Decode attention tensor is not f32 aligned");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
