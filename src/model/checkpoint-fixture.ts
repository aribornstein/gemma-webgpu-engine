interface CheckpointTensorMetadata {
  dtype: "float16" | "float32" | "uint8" | "uint32";
  arrayType: "Float32Array" | "Uint32Array" | "Uint16Array" | "Uint8Array";
  shape: number[];
  sha256: string;
}

interface CheckpointMetadata {
  schemaVersion: 1;
  tensorFile: string;
  tensorFileSha256: string;
  tensors: Record<string, CheckpointTensorMetadata>;
  [key: string]: unknown;
}

interface SafeTensorHeader {
  dtype: "F16" | "F32" | "U8" | "U32";
  shape: number[];
  data_offsets: [number, number];
}

export interface LoadedCheckpointFixture {
  metadata: CheckpointMetadata;
  metadataSha256: string;
  tensorFileSha256: string;
  tensors: Map<string, Uint8Array>;
}

const SAFE_DTYPES = {
  float16: { header: "F16", bytes: 2 },
  float32: { header: "F32", bytes: 4 },
  uint8: { header: "U8", bytes: 1 },
  uint32: { header: "U32", bytes: 4 },
} as const;

export async function loadCheckpointFixture(
  metadataUrl: string,
  expectedMetadataSha256: string,
  expectedTensorFileSha256: string,
): Promise<LoadedCheckpointFixture> {
  const metadataResponse = await fetch(metadataUrl);
  if (!metadataResponse.ok) {
    throw new Error(`Failed to load checkpoint metadata: ${metadataResponse.status}`);
  }
  const metadataBytes = new Uint8Array(await metadataResponse.arrayBuffer());
  const metadataSha256 = await sha256(metadataBytes);
  if (metadataSha256 !== expectedMetadataSha256) {
    throw new Error(`Checkpoint metadata hash mismatch: ${metadataSha256}`);
  }
  const metadata = JSON.parse(new TextDecoder().decode(metadataBytes)) as CheckpointMetadata;
  if (
    metadata.schemaVersion !== 1 ||
    typeof metadata.tensorFile !== "string" ||
    metadata.tensorFileSha256 !== expectedTensorFileSha256 ||
    typeof metadata.tensors !== "object" ||
    metadata.tensors === null
  ) {
    throw new Error("Checkpoint metadata contract mismatch");
  }

  const tensorUrl = new URL(metadata.tensorFile, metadataUrl).href;
  const tensorResponse = await fetch(tensorUrl);
  if (!tensorResponse.ok) {
    throw new Error(`Failed to load checkpoint tensors: ${tensorResponse.status}`);
  }
  const tensorBytes = new Uint8Array(await tensorResponse.arrayBuffer());
  const tensorFileSha256 = await sha256(tensorBytes);
  if (tensorFileSha256 !== expectedTensorFileSha256) {
    throw new Error(`Checkpoint tensor hash mismatch: ${tensorFileSha256}`);
  }
  if (tensorBytes.byteLength < 8) throw new Error("Checkpoint tensor prefix is missing");
  const headerLength = Number(
    new DataView(tensorBytes.buffer, tensorBytes.byteOffset, 8).getBigUint64(0, true),
  );
  const dataStart = 8 + headerLength;
  if (!Number.isSafeInteger(headerLength) || dataStart > tensorBytes.byteLength) {
    throw new Error("Checkpoint tensor header length is invalid");
  }
  const header = JSON.parse(
    new TextDecoder().decode(tensorBytes.subarray(8, dataStart)).trim(),
  ) as Record<string, SafeTensorHeader>;
  if (
    Object.keys(header).length !== Object.keys(metadata.tensors).length ||
    Object.keys(header).some((name) => !(name in metadata.tensors))
  ) {
    throw new Error("Checkpoint tensor names do not match metadata");
  }

  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(metadata.tensors)) {
    const tensor = header[name];
    const dtype = SAFE_DTYPES[expected.dtype];
    const elements = expected.shape.reduce((product, dimension) => product * dimension, 1);
    if (
      !tensor || tensor.dtype !== dtype.header ||
      tensor.shape.join(",") !== expected.shape.join(",") ||
      !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2 ||
      tensor.data_offsets[1] - tensor.data_offsets[0] !== elements * dtype.bytes ||
      tensor.data_offsets[0] < 0 || tensor.data_offsets[1] < tensor.data_offsets[0] ||
      dataStart + tensor.data_offsets[1] > tensorBytes.byteLength
    ) {
      throw new Error(`Checkpoint tensor contract mismatch for ${name}`);
    }
    const payload = tensorBytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const payloadSha256 = await sha256(payload);
    if (payloadSha256 !== expected.sha256) {
      throw new Error(`Checkpoint tensor payload mismatch for ${name}: ${payloadSha256}`);
    }
    tensors.set(name, payload);
  }
  return { metadata, metadataSha256, tensorFileSha256, tensors };
}

export function checkpointFloat32(
  fixture: LoadedCheckpointFixture,
  name: string,
): Float32Array {
  return convert(fixture, name, "Float32Array", 4, (view, offset) =>
    view.getFloat32(offset, true));
}

export function checkpointUint32(
  fixture: LoadedCheckpointFixture,
  name: string,
): Uint32Array {
  return convert(fixture, name, "Uint32Array", 4, (view, offset) =>
    view.getUint32(offset, true));
}

export function checkpointUint16(
  fixture: LoadedCheckpointFixture,
  name: string,
): Uint16Array {
  return convert(fixture, name, "Uint16Array", 2, (view, offset) =>
    view.getUint16(offset, true));
}

function convert<T extends Float32Array | Uint32Array | Uint16Array>(
  fixture: LoadedCheckpointFixture,
  name: string,
  expectedArrayType: CheckpointTensorMetadata["arrayType"],
  bytesPerElement: number,
  read: (view: DataView, offset: number) => number,
): T {
  const bytes = fixture.tensors.get(name);
  const metadata = fixture.metadata.tensors[name];
  if (!bytes || metadata?.arrayType !== expectedArrayType) {
    throw new Error(`Checkpoint tensor ${name} is not ${expectedArrayType}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = Array.from(
    { length: bytes.byteLength / bytesPerElement },
    (_, index) => read(view, index * bytesPerElement),
  );
  const Constructor = expectedArrayType === "Float32Array"
    ? Float32Array
    : expectedArrayType === "Uint32Array"
      ? Uint32Array
      : Uint16Array;
  return new Constructor(values) as T;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}