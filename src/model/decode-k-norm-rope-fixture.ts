import { modelAssetUrl } from "./model-assets";

const FIXTURE_URL = modelAssetUrl(
  "gemma-4-e2b/operators/decode-k-norm-rope-layer0.safetensors",
);
const CONTAINER_SHA256 = "b0a44ef55d5dc0d9a827d7f4171fe36c15a6f37a4f74be98bae701708e88a374";
const SOURCE_CAPTURE_SHA256 = "adc161c83071906c53ecf521d4c9fa140d01f729e5ecb6017d06860592306106";

const EXPECTED = {
  input: {
    dtype: "F32",
    shape: [256],
    sha256: "d7fc62c5b4f34fbcb6ec4ac1a73d1dfa7c20ed53683ac3bb86e263b246e61bb8",
  },
  weight: {
    dtype: "F32",
    shape: [256],
    sha256: "3945bd6a02a7ff0e9dc52f6008c989907a7e91e6a8ded0de5c588fed739a8970",
  },
  cosine: {
    dtype: "F32",
    shape: [128],
    sha256: "0a8d34ce643de6612a8725b36cb6599f3de2bb956fbd3c96f7577014c9707141",
  },
  sine: {
    dtype: "F32",
    shape: [128],
    sha256: "078b5542f3148c0eeb381bd8be0bec9b2770f366f0466093c984e659807b5132",
  },
  output: {
    dtype: "F32",
    shape: [256],
    sha256: "370fb08cb473409f416d30a7df22b4cb2752ab0c2703f563fab5d2fcaeff7d94",
  },
} as const;

interface TensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface DecodeKNormRopeFixture {
  input: Float32Array;
  weight: Float32Array;
  cosine: Float32Array;
  sine: Float32Array;
  expectedOutput: Float32Array;
  headDim: 256;
  halfDim: 128;
  heads: 1;
  epsilon: 0.000001;
  position: 10;
  artifactUrl: string;
  artifactSha256: string;
  sourceCaptureSha256: string;
}

let fixturePromise: Promise<DecodeKNormRopeFixture> | null = null;

export function loadDecodeKNormRopeFixture(): Promise<DecodeKNormRopeFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

async function loadFixture(): Promise<DecodeKNormRopeFixture> {
  const response = await fetch(FIXTURE_URL);
  if (!response.ok) {
    throw new Error(`Failed to load DecodeQkNormRope fixture: ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const artifactSha256 = await sha256(bytes);
  if (artifactSha256 !== CONTAINER_SHA256) {
    throw new Error(`DecodeQkNormRope fixture hash mismatch: ${artifactSha256}`);
  }
  const tensors = await readExpectedTensors(bytes);
  return {
    input: float32LittleEndian(requiredTensor(tensors, "input")),
    weight: float32LittleEndian(requiredTensor(tensors, "weight")),
    cosine: float32LittleEndian(requiredTensor(tensors, "cosine")),
    sine: float32LittleEndian(requiredTensor(tensors, "sine")),
    expectedOutput: float32LittleEndian(requiredTensor(tensors, "output")),
    headDim: 256,
    halfDim: 128,
    heads: 1,
    epsilon: 0.000001,
    position: 10,
    artifactUrl: FIXTURE_URL,
    artifactSha256,
    sourceCaptureSha256: SOURCE_CAPTURE_SHA256,
  };
}

async function readExpectedTensors(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  if (bytes.byteLength < 8) throw new Error("DecodeQkNormRope fixture is missing its prefix");
  const headerBytes = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerBytes;
  if (!Number.isSafeInteger(headerBytes) || dataStart > bytes.byteLength) {
    throw new Error("DecodeQkNormRope fixture has an invalid header length");
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, dataStart)).trim(),
  ) as Record<string, unknown>;
  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(EXPECTED)) {
    const tensor = parseTensorHeader(header[name], name);
    if (tensor.dtype !== expected.dtype || tensor.shape.join(",") !== expected.shape.join(",")) {
      throw new Error(`DecodeQkNormRope fixture metadata mismatch for ${name}`);
    }
    const payload = bytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const actualSha256 = await sha256(payload);
    if (actualSha256 !== expected.sha256) {
      throw new Error(`DecodeQkNormRope tensor hash mismatch for ${name}: ${actualSha256}`);
    }
    tensors.set(name, payload);
  }
  return tensors;
}

function parseTensorHeader(value: unknown, name: string): TensorHeader {
  if (typeof value !== "object" || value === null) {
    throw new Error(`DecodeQkNormRope fixture is missing tensor ${name}`);
  }
  const tensor = value as Partial<TensorHeader>;
  if (
    typeof tensor.dtype !== "string" || !Array.isArray(tensor.shape) ||
    !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2
  ) {
    throw new Error(`DecodeQkNormRope fixture has an invalid header for ${name}`);
  }
  return tensor as TensorHeader;
}

function requiredTensor(tensors: Map<string, Uint8Array>, name: string): Uint8Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`DecodeQkNormRope fixture is missing payload ${name}`);
  return tensor;
}

function float32LittleEndian(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("DecodeQkNormRope tensor is not f32 aligned");
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
