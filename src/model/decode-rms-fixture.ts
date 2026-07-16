import { modelAssetUrl } from "./model-assets";

const FIXTURE_URL = modelAssetUrl(
  "gemma-4-e2b/operators/decode-rms-srq-layer0.safetensors",
);
const CONTAINER_SHA256 = "75edf39811df47143afcf92fd8e64931820eae808e9a6a11a2b57e4464202c36";
const SOURCE_FIXTURE_SHA256 = "e511c4e9e201266a9f32e06a9672a6377aacdd32d44ecb143e6e000a8cc3f03e";
const SOURCE_CAPTURE_SHA256 = "78dcbadb59abdb04d51facaa1af5674fc40b7260503fcc900239cd00890f1ae9";

const EXPECTED = {
  hidden: {
    dtype: "F32",
    shape: [1536],
    sha256: "4b7c92982f60cc985f150bdbb22a28a8e276bf516edd2b6a24c284fb810ed6a2",
  },
  weight: {
    dtype: "F32",
    shape: [1536],
    sha256: "9232aeea8b9e2234079a69227143f63eb07b358755c4c325f5709ac91ce6e91d",
  },
  output: {
    dtype: "F32",
    shape: [1536],
    sha256: "4fc6cc0893feceddf48e1c8f89a0e152d067ac5f5b655d3cef7853d961be66ae",
  },
  sum_a: {
    dtype: "F32",
    shape: [1],
    sha256: "fd8d3caba9e940d6a742a81b0bfa0faf6af8f6ec8a4fa21fca2ab54d7b117350",
  },
} as const;

interface TensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface DecodeRmsFixture {
  hidden: Float32Array;
  weight: Float32Array;
  expectedOutput: Float32Array;
  expectedSum: Float32Array;
  hiddenSize: 1536;
  epsilon: 0.000001;
  inputScale: 0.6088034510612488;
  artifactUrl: string;
  artifactSha256: string;
  sourceFixtureSha256: string;
  sourceCaptureSha256: string;
}

let fixturePromise: Promise<DecodeRmsFixture> | null = null;

export function loadDecodeRmsFixture(): Promise<DecodeRmsFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

async function loadFixture(): Promise<DecodeRmsFixture> {
  const response = await fetch(FIXTURE_URL);
  if (!response.ok) throw new Error(`Failed to load DecodeRmsSrq fixture: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const artifactSha256 = await sha256(bytes);
  if (artifactSha256 !== CONTAINER_SHA256) {
    throw new Error(`DecodeRmsSrq fixture hash mismatch: ${artifactSha256}`);
  }
  const tensors = await readExpectedTensors(bytes);
  return {
    hidden: float32LittleEndian(requiredTensor(tensors, "hidden")),
    weight: float32LittleEndian(requiredTensor(tensors, "weight")),
    expectedOutput: float32LittleEndian(requiredTensor(tensors, "output")),
    expectedSum: float32LittleEndian(requiredTensor(tensors, "sum_a")),
    hiddenSize: 1536,
    epsilon: 0.000001,
    inputScale: 0.6088034510612488,
    artifactUrl: FIXTURE_URL,
    artifactSha256,
    sourceFixtureSha256: SOURCE_FIXTURE_SHA256,
    sourceCaptureSha256: SOURCE_CAPTURE_SHA256,
  };
}

async function readExpectedTensors(bytes: Uint8Array): Promise<Map<string, Uint8Array>> {
  if (bytes.byteLength < 8) throw new Error("DecodeRmsSrq fixture is missing its prefix");
  const headerBytes = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerBytes;
  if (!Number.isSafeInteger(headerBytes) || dataStart > bytes.byteLength) {
    throw new Error("DecodeRmsSrq fixture has an invalid header length");
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, dataStart)).trim(),
  ) as Record<string, unknown>;
  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(EXPECTED)) {
    const tensor = parseTensorHeader(header[name], name);
    if (tensor.dtype !== expected.dtype || tensor.shape.join(",") !== expected.shape.join(",")) {
      throw new Error(`DecodeRmsSrq fixture metadata mismatch for ${name}`);
    }
    const payload = bytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const actualSha256 = await sha256(payload);
    if (actualSha256 !== expected.sha256) {
      throw new Error(`DecodeRmsSrq tensor hash mismatch for ${name}: ${actualSha256}`);
    }
    tensors.set(name, payload);
  }
  return tensors;
}

function parseTensorHeader(value: unknown, name: string): TensorHeader {
  if (typeof value !== "object" || value === null) {
    throw new Error(`DecodeRmsSrq fixture is missing tensor ${name}`);
  }
  const tensor = value as Partial<TensorHeader>;
  if (
    typeof tensor.dtype !== "string" || !Array.isArray(tensor.shape) ||
    !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2
  ) {
    throw new Error(`DecodeRmsSrq fixture has an invalid header for ${name}`);
  }
  return tensor as TensorHeader;
}

function requiredTensor(tensors: Map<string, Uint8Array>, name: string): Uint8Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`DecodeRmsSrq fixture is missing payload ${name}`);
  return tensor;
}

function float32LittleEndian(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("DecodeRmsSrq tensor is not f32 aligned");
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