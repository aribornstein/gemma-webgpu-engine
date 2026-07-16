import type { QatLinearFixture } from "../reference/qat-linear";
import { modelAssetUrl } from "./model-assets";

const OPERATOR = "model.language_model.layers.0.self_attn.q_proj";
const FIXTURE_URL = modelAssetUrl("gemma-4-e2b/operators/layer0-q-proj.safetensors");
const CONTAINER_SHA256 = "932bfa1d84087dba4ef2104a801431a9b8c9a0fd7a25f7ff65d83bea1d062be6";
const GOLDEN_URL = modelAssetUrl(
  "gemma-4-e2b/operators/layer0-q-proj-golden.safetensors",
);
const GOLDEN_SHA256 = "e9ef6a3b477cda98572611840f6a87965a9460221f4bc2b5473f535b50b876f2";
const CAPTURE_URL = modelAssetUrl(
  "gemma-4-e2b/operators/layer0-q-proj-buza-decode.json",
);
const CAPTURE_SHA256 = "78dcbadb59abdb04d51facaa1af5674fc40b7260503fcc900239cd00890f1ae9";
const CAPTURE_COMMIT = "158f16ae0f672943ca304d59c47c8e3a264e399e";
const CAPTURE_BUNDLE_SHA256 = "0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62";
const QKV_FIXTURE_URL = modelAssetUrl("gemma-4-e2b/operators/layer0-qkv.safetensors");
const QKV_CONTAINER_SHA256 = "63482ab46577cc82b15879a8db0b0fea4515fc690741c34f4e47fb2d6faab1e3";

const EXPECTED = {
  [`${OPERATOR}.input_activation_scale`]: {
    dtype: "F32",
    shape: [],
    sha256: "610be057354d89a641e99b1f49669b4ac947a505d875671b01b65d8a7c6f3baa",
  },
  [`${OPERATOR}.output_activation_scale`]: {
    dtype: "F32",
    shape: [],
    sha256: "6e10b5e4202615ca485fbcda79c2436b179bee82238a2c1e1165257f5356f3c8",
  },
  [`${OPERATOR}.weight_scale`]: {
    dtype: "F32",
    shape: [2048, 1],
    sha256: "aed60558da8d6439784db2e0bdb20e115739d80c0e72d68d4818b8f3c71906c0",
  },
  [`${OPERATOR}.weight`]: {
    dtype: "U8",
    shape: [2048, 768],
    sha256: "761063d579e071903e64f9d4eca1aa3fcf236eddf345a2d81d2e1d62fdea9b7c",
  },
} as const;

const GOLDEN_EXPECTED = {
  q_input: {
    dtype: "F32",
    shape: [1536],
    sha256: "e17523a27908584b70a360219a366a47292fccdb286c30ca987666b4db86d8b0",
  },
  q_input_srq: {
    dtype: "F32",
    shape: [1536],
    sha256: "fb008657ed852850bdc44ee210e6eb1f25fb30391ddb7c97f236bb5809377839",
  },
  q_output: {
    dtype: "F32",
    shape: [2048],
    sha256: "b2dfb8d5a199cdfec5aa93c4056f2c66724c24e8a553603808e7b0a33e1326b2",
  },
} as const;

const QKV_EXPECTED = {
  "model.language_model.layers.0.self_attn.q_proj.input_activation_scale": {
    dtype: "F32", shape: [], sha256: "610be057354d89a641e99b1f49669b4ac947a505d875671b01b65d8a7c6f3baa",
  },
  "model.language_model.layers.0.self_attn.q_proj.output_activation_scale": {
    dtype: "F32", shape: [], sha256: "6e10b5e4202615ca485fbcda79c2436b179bee82238a2c1e1165257f5356f3c8",
  },
  "model.language_model.layers.0.self_attn.q_proj.weight_scale": {
    dtype: "F32", shape: [2048, 1], sha256: "aed60558da8d6439784db2e0bdb20e115739d80c0e72d68d4818b8f3c71906c0",
  },
  "model.language_model.layers.0.self_attn.q_proj.weight": {
    dtype: "U8", shape: [2048, 768], sha256: "761063d579e071903e64f9d4eca1aa3fcf236eddf345a2d81d2e1d62fdea9b7c",
  },
  "model.language_model.layers.0.self_attn.k_proj.input_activation_scale": {
    dtype: "F32", shape: [], sha256: "610be057354d89a641e99b1f49669b4ac947a505d875671b01b65d8a7c6f3baa",
  },
  "model.language_model.layers.0.self_attn.k_proj.output_activation_scale": {
    dtype: "F32", shape: [], sha256: "dbe8676730143af9fdae8afa863daf09635d557bf06c48b6d991b8f9224e1ef6",
  },
  "model.language_model.layers.0.self_attn.k_proj.weight_scale": {
    dtype: "F32", shape: [256, 1], sha256: "6dcf1d580123e38ba3762a6b86c31b6a6c974ec02599911e5709a4d51df2adc7",
  },
  "model.language_model.layers.0.self_attn.k_proj.weight": {
    dtype: "U8", shape: [256, 768], sha256: "aab1f9f365d0e33c5e2f751c7d9ce3ccc147964f64416e1c69a8f9575ce145cc",
  },
  "model.language_model.layers.0.self_attn.v_proj.input_activation_scale": {
    dtype: "F32", shape: [], sha256: "610be057354d89a641e99b1f49669b4ac947a505d875671b01b65d8a7c6f3baa",
  },
  "model.language_model.layers.0.self_attn.v_proj.output_activation_scale": {
    dtype: "F32", shape: [], sha256: "dbe8676730143af9fdae8afa863daf09635d557bf06c48b6d991b8f9224e1ef6",
  },
  "model.language_model.layers.0.self_attn.v_proj.weight_scale": {
    dtype: "F32", shape: [256, 1], sha256: "4a1606c3ec0cbcaf4dd89e18cc0b7f1a43c98ac213124f5d68c2834409740bfa",
  },
  "model.language_model.layers.0.self_attn.v_proj.weight": {
    dtype: "U8", shape: [256, 768], sha256: "47f81fbf9cd680829baa7df3afbd45c8c1dfaf55705c9efd821770b3fed68b85",
  },
} as const;

interface TensorHeader {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface RealQatLinearFixture {
  fixture: QatLinearFixture;
  expectedOutput: Float32Array;
  artifactUrl: string;
  containerSha256: string;
  referenceUrl: string;
  referenceSha256: string;
  inputActivationScale: number;
  outputActivationScale: number;
}

export interface CapturedQatQkvFixture {
  input: Float32Array;
  inputSum: Float32Array;
  packedWeights: Uint32Array;
  qPackedWeights: Uint32Array;
  kPackedWeights: Uint32Array;
  vPackedWeights: Uint32Array;
  rowScales: Float32Array;
  expectedQ: Float32Array;
  expectedK: Float32Array;
  expectedV: Float32Array;
  outputScales: Float32Array;
  artifactUrl: string;
  artifactSha256: string;
  referenceUrl: string;
  referenceSha256: string;
}

let fixturePromise: Promise<RealQatLinearFixture> | null = null;
let capturedFixturePromise: Promise<RealQatLinearFixture> | null = null;
let capturedQkvFixturePromise: Promise<CapturedQatQkvFixture> | null = null;

export function loadRealQatLinearFixture(): Promise<RealQatLinearFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

export function loadCapturedQatLinearFixture(): Promise<RealQatLinearFixture> {
  capturedFixturePromise ??= loadCapturedFixture().catch((error) => {
    capturedFixturePromise = null;
    throw error;
  });
  return capturedFixturePromise;
}

export function loadCapturedQatQkvFixture(): Promise<CapturedQatQkvFixture> {
  capturedQkvFixturePromise ??= loadCapturedQkvFixture().catch((error) => {
    capturedQkvFixturePromise = null;
    throw error;
  });
  return capturedQkvFixturePromise;
}

async function loadCapturedQkvFixture(): Promise<CapturedQatQkvFixture> {
  const [fixtureResponse, captureResponse] = await Promise.all([
    fetch(QKV_FIXTURE_URL),
    fetch(CAPTURE_URL),
  ]);
  if (!fixtureResponse.ok) throw new Error(`Failed to load QKV fixture: ${fixtureResponse.status}`);
  if (!captureResponse.ok) throw new Error(`Failed to load decode capture: ${captureResponse.status}`);

  const [fixtureBytes, captureBytes] = await Promise.all([
    fixtureResponse.arrayBuffer().then((buffer) => new Uint8Array(buffer)),
    captureResponse.arrayBuffer().then((buffer) => new Uint8Array(buffer)),
  ]);
  const [artifactSha256, referenceSha256] = await Promise.all([
    sha256(fixtureBytes),
    sha256(captureBytes),
  ]);
  if (artifactSha256 !== QKV_CONTAINER_SHA256) {
    throw new Error(`QKV fixture container hash mismatch: ${artifactSha256}`);
  }
  if (referenceSha256 !== CAPTURE_SHA256) {
    throw new Error(`Decode capture hash mismatch: ${referenceSha256}`);
  }

  const tensors = await readExpectedTensors(fixtureBytes, QKV_EXPECTED, "QKV fixture");
  const capture = JSON.parse(new TextDecoder().decode(captureBytes)) as Record<string, unknown>;
  const source = requiredRecord(capture.source, "capture source");
  if (source.upstreamCommit !== CAPTURE_COMMIT || source.bundleSha256 !== CAPTURE_BUNDLE_SHA256) {
    throw new Error("Decode capture source identity mismatch");
  }
  const projection = requiredRecord(capture.projection, "capture projection");
  const qPrefix = "model.language_model.layers.0.self_attn.q_proj";
  const kPrefix = "model.language_model.layers.0.self_attn.k_proj";
  const vPrefix = "model.language_model.layers.0.self_attn.v_proj";
  const inputScales = [qPrefix, kPrefix, vPrefix].map((prefix) =>
    float32Scalar(requiredTensor(tensors, `${prefix}.input_activation_scale`))
  );
  const outputScales = Float32Array.from([qPrefix, kPrefix, vPrefix], (prefix) =>
    float32Scalar(requiredTensor(tensors, `${prefix}.output_activation_scale`))
  );
  if (
    projection.bits !== 4 || projection.inFeatures !== 1536 ||
    projection.qOut !== 2048 || projection.kvOut !== 256 ||
    inputScales.some((scale) => scale !== projection.inputScale) ||
    outputScales[0] !== projection.qOutputScale ||
    outputScales[1] !== projection.kOutputScale ||
    outputScales[2] !== projection.vOutputScale
  ) {
    throw new Error("Decode capture QKV projection metadata mismatch");
  }

  const captureTensors = requiredRecord(capture.tensors, "capture tensors");
  const qPackedWeights = uint32LittleEndian(requiredTensor(tensors, `${qPrefix}.weight`));
  const kPackedWeights = uint32LittleEndian(requiredTensor(tensors, `${kPrefix}.weight`));
  const vPackedWeights = uint32LittleEndian(requiredTensor(tensors, `${vPrefix}.weight`));
  return {
    input: await capturedFloat32Tensor(captureTensors.input, "input", [1, 1536], "4fc6cc0893feceddf48e1c8f89a0e152d067ac5f5b655d3cef7853d961be66ae"),
    inputSum: await capturedFloat32Tensor(captureTensors.sumA, "sumA", [1], "fd8d3caba9e940d6a742a81b0bfa0faf6af8f6ec8a4fa21fca2ab54d7b117350"),
    packedWeights: concatenateUint32([qPackedWeights, kPackedWeights, vPackedWeights]),
    qPackedWeights,
    kPackedWeights,
    vPackedWeights,
    rowScales: concatenateFloat32([
      float32LittleEndian(requiredTensor(tensors, `${qPrefix}.weight_scale`)),
      float32LittleEndian(requiredTensor(tensors, `${kPrefix}.weight_scale`)),
      float32LittleEndian(requiredTensor(tensors, `${vPrefix}.weight_scale`)),
    ]),
    expectedQ: await capturedFloat32Tensor(captureTensors.q, "q", [1, 2048], "ab887bc85137320455da9e3a9b5b8e121bb19dced68326cfa1434f82b951482b"),
    expectedK: await capturedFloat32Tensor(captureTensors.k, "k", [1, 256], "d7fc62c5b4f34fbcb6ec4ac1a73d1dfa7c20ed53683ac3bb86e263b246e61bb8"),
    expectedV: await capturedFloat32Tensor(captureTensors.v, "v", [1, 256], "03bfb63ecb754de7d74dc97316bc813a856446873940073a63f754d609d6be36"),
    outputScales,
    artifactUrl: QKV_FIXTURE_URL,
    artifactSha256,
    referenceUrl: CAPTURE_URL,
    referenceSha256,
  };
}

async function loadCapturedFixture(): Promise<RealQatLinearFixture> {
  const [loaded, response] = await Promise.all([
    loadRealQatLinearFixture(),
    fetch(CAPTURE_URL),
  ]);
  if (!response.ok) throw new Error(`Failed to load Buza decode capture: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const referenceSha256 = await sha256(bytes);
  if (referenceSha256 !== CAPTURE_SHA256) {
    throw new Error(`Buza decode capture hash mismatch: ${referenceSha256}`);
  }
  const capture = JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
  const source = requiredRecord(capture.source, "capture source");
  if (
    source.upstreamCommit !== CAPTURE_COMMIT ||
    source.bundleSha256 !== CAPTURE_BUNDLE_SHA256
  ) {
    throw new Error("Buza decode capture source identity mismatch");
  }
  const projection = requiredRecord(capture.projection, "capture projection");
  if (
    projection.bits !== 4 ||
    projection.inFeatures !== 1536 ||
    projection.qOut !== 2048 ||
    projection.inputScale !== loaded.inputActivationScale ||
    projection.qOutputScale !== loaded.outputActivationScale
  ) {
    throw new Error("Buza decode capture projection metadata mismatch");
  }
  const tensors = requiredRecord(capture.tensors, "capture tensors");
  const input = await capturedFloat32Tensor(
    tensors.input,
    "input",
    [1, 1536],
    "4fc6cc0893feceddf48e1c8f89a0e152d067ac5f5b655d3cef7853d961be66ae",
  );
  const inputSum = await capturedFloat32Tensor(
    tensors.sumA,
    "sumA",
    [1],
    "fd8d3caba9e940d6a742a81b0bfa0faf6af8f6ec8a4fa21fca2ab54d7b117350",
  );
  const expectedOutput = await capturedFloat32Tensor(
    tensors.q,
    "q",
    [1, 2048],
    "ab887bc85137320455da9e3a9b5b8e121bb19dced68326cfa1434f82b951482b",
  );
  return {
    ...loaded,
    fixture: {
      ...loaded.fixture,
      input,
      inputSum,
      emulateBfloat16: false,
    },
    expectedOutput,
    referenceUrl: CAPTURE_URL,
    referenceSha256,
  };
}

async function loadFixture(): Promise<RealQatLinearFixture> {
  const [response, goldenResponse] = await Promise.all([fetch(FIXTURE_URL), fetch(GOLDEN_URL)]);
  if (!response.ok) throw new Error(`Failed to load QAT fixture: ${response.status}`);
  if (!goldenResponse.ok) {
    throw new Error(`Failed to load QAT golden reference: ${goldenResponse.status}`);
  }
  const [bytes, goldenBytes] = await Promise.all([
    response.arrayBuffer().then((buffer) => new Uint8Array(buffer)),
    goldenResponse.arrayBuffer().then((buffer) => new Uint8Array(buffer)),
  ]);
  const containerSha256 = await sha256(bytes);
  if (containerSha256 !== CONTAINER_SHA256) {
    throw new Error(`QAT fixture container hash mismatch: ${containerSha256}`);
  }
  const referenceSha256 = await sha256(goldenBytes);
  if (referenceSha256 !== GOLDEN_SHA256) {
    throw new Error(`QAT golden reference hash mismatch: ${referenceSha256}`);
  }
  const [tensors, goldenTensors] = await Promise.all([
    readExpectedTensors(bytes, EXPECTED, "QAT fixture"),
    readExpectedTensors(goldenBytes, GOLDEN_EXPECTED, "QAT golden reference"),
  ]);

  const rowScaleBytes = requiredTensor(tensors, `${OPERATOR}.weight_scale`);
  const weightBytes = requiredTensor(tensors, `${OPERATOR}.weight`);
  const inputActivationScale = float32Scalar(
    requiredTensor(tensors, `${OPERATOR}.input_activation_scale`),
  );
  const outputActivationScale = float32Scalar(
    requiredTensor(tensors, `${OPERATOR}.output_activation_scale`),
  );
  return {
    fixture: {
      input: float32LittleEndian(requiredTensor(goldenTensors, "q_input")),
      packedWeights: uint32LittleEndian(weightBytes),
      rowScales: float32LittleEndian(rowScaleBytes),
      inFeatures: 1536,
      outFeatures: 2048,
      inputActivationScale,
      outputActivationScale,
      emulateBfloat16: true,
    },
    expectedOutput: float32LittleEndian(requiredTensor(goldenTensors, "q_output")),
    artifactUrl: FIXTURE_URL,
    containerSha256,
    referenceUrl: GOLDEN_URL,
    referenceSha256,
    inputActivationScale,
    outputActivationScale,
  };
}

async function readExpectedTensors(
  bytes: Uint8Array,
  expectedTensors: Record<string, { dtype: string; shape: readonly number[]; sha256: string }>,
  label: string,
): Promise<Map<string, Uint8Array>> {
  if (bytes.byteLength < 8) throw new Error(`${label} is missing its safetensors prefix`);
  const headerBytes = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const dataStart = 8 + headerBytes;
  if (!Number.isSafeInteger(headerBytes) || dataStart > bytes.byteLength) {
    throw new Error(`${label} has an invalid safetensors header length`);
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(8, dataStart)).trim(),
  ) as Record<string, unknown>;
  const tensors = new Map<string, Uint8Array>();
  for (const [name, expected] of Object.entries(expectedTensors)) {
    const tensor = parseTensorHeader(header[name], name);
    if (tensor.dtype !== expected.dtype || tensor.shape.join(",") !== expected.shape.join(",")) {
      throw new Error(`${label} metadata mismatch for ${name}`);
    }
    const payload = bytes.subarray(
      dataStart + tensor.data_offsets[0],
      dataStart + tensor.data_offsets[1],
    );
    const actualHash = await sha256(payload);
    if (actualHash !== expected.sha256) {
      throw new Error(`${label} tensor hash mismatch for ${name}: ${actualHash}`);
    }
    tensors.set(name, payload);
  }
  return tensors;
}

function parseTensorHeader(value: unknown, name: string): TensorHeader {
  if (typeof value !== "object" || value === null) {
    throw new Error(`QAT fixture is missing tensor ${name}`);
  }
  const tensor = value as Partial<TensorHeader>;
  if (typeof tensor.dtype !== "string" || !Array.isArray(tensor.shape) ||
      !Array.isArray(tensor.data_offsets) || tensor.data_offsets.length !== 2) {
    throw new Error(`QAT fixture has an invalid header for ${name}`);
  }
  return tensor as TensorHeader;
}

function requiredTensor(tensors: Map<string, Uint8Array>, name: string): Uint8Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`QAT fixture is missing payload ${name}`);
  return tensor;
}

function uint32LittleEndian(bytes: Uint8Array): Uint32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("Packed weight bytes are not u32 aligned");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Uint32Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  );
}

function float32LittleEndian(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("Scale bytes are not f32 aligned");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Float32Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}

function float32Scalar(bytes: Uint8Array): number {
  if (bytes.byteLength !== 4) throw new Error("Activation scale must be one f32");
  return new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);
}

function concatenateFloat32(arrays: Float32Array[]): Float32Array {
  const result = new Float32Array(arrays.reduce((length, array) => length + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    result.set(array, offset);
    offset += array.length;
  }
  return result;
}

function concatenateUint32(arrays: Uint32Array[]): Uint32Array {
  const result = new Uint32Array(arrays.reduce((length, array) => length + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    result.set(array, offset);
    offset += array.length;
  }
  return result;
}

function requiredRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

async function capturedFloat32Tensor(
  value: unknown,
  name: string,
  expectedShape: number[],
  expectedSha256: string,
): Promise<Float32Array> {
  const tensor = requiredRecord(value, `capture tensor ${name}`);
  if (
    tensor.dtype !== "float32" ||
    !Array.isArray(tensor.shape) ||
    tensor.shape.join(",") !== expectedShape.join(",") ||
    tensor.sha256 !== expectedSha256 ||
    !Array.isArray(tensor.values)
  ) {
    throw new Error(`capture tensor ${name} metadata mismatch`);
  }
  const values = Float32Array.from(tensor.values as number[]);
  if (values.length !== expectedShape.reduce((product, size) => product * size, 1)) {
    throw new Error(`capture tensor ${name} value count mismatch`);
  }
  const actualSha256 = await sha256(new Uint8Array(values.buffer));
  if (actualSha256 !== expectedSha256) {
    throw new Error(`capture tensor ${name} payload hash mismatch: ${actualSha256}`);
  }
  return values;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}