import type {
  CachedTensorDescriptor,
  CachedTensorPayload,
} from "./cached-safetensors";

const MODEL_REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";
const MODEL_FILE_SIZE = 2_458_111_846;
const HEADER_BYTES = 375_392;
const DATA_START = 375_400;
const TENSOR_COUNT = 2_780;

export const GEMMA_PINNED_SAFETENSORS_URL =
  `https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/` +
  `${MODEL_REVISION}/model.safetensors`;

interface RawTensorDescriptor {
  dtype?: unknown;
  shape?: unknown;
  data_offsets?: unknown;
}

export class PinnedSafetensorsSource {
  readonly descriptors: ReadonlyMap<string, CachedTensorDescriptor>;
  private readonly sourceUrl: string;

  private constructor(
    sourceUrl: string,
    descriptors: ReadonlyMap<string, CachedTensorDescriptor>,
  ) {
    this.sourceUrl = sourceUrl;
    this.descriptors = descriptors;
  }

  static async open(sourceUrl = GEMMA_PINNED_SAFETENSORS_URL): Promise<PinnedSafetensorsSource> {
    const header = await fetchExactRange(sourceUrl, 0, DATA_START);
    const prefix = Number(new DataView(
      header.buffer,
      header.byteOffset,
      8,
    ).getBigUint64(0, true));
    if (prefix !== HEADER_BYTES) {
      throw new Error(`Pinned safetensors header length mismatch: ${prefix}`);
    }
    const parsed = JSON.parse(new TextDecoder().decode(header.subarray(8)).trim()) as
      Record<string, RawTensorDescriptor>;
    const entries = Object.entries(parsed).filter(([name]) => name !== "__metadata__");
    if (entries.length !== TENSOR_COUNT) {
      throw new Error(`Pinned safetensors tensor count mismatch: ${entries.length}`);
    }
    const descriptors = new Map<string, CachedTensorDescriptor>();
    for (const [name, raw] of entries) {
      if (typeof raw.dtype !== "string" || !Array.isArray(raw.shape) ||
          !raw.shape.every((dimension) => Number.isInteger(dimension) && dimension >= 0) ||
          !Array.isArray(raw.data_offsets) || raw.data_offsets.length !== 2 ||
          !raw.data_offsets.every(Number.isInteger)) {
        throw new Error(`Pinned safetensors descriptor is invalid for ${name}`);
      }
      const [relativeBegin, relativeEnd] = raw.data_offsets as [number, number];
      const begin = DATA_START + relativeBegin;
      const end = DATA_START + relativeEnd;
      if (relativeBegin < 0 || relativeEnd < relativeBegin || end > MODEL_FILE_SIZE) {
        throw new Error(`Pinned safetensors range is invalid for ${name}`);
      }
      descriptors.set(name, {
        name,
        dtype: raw.dtype,
        shape: raw.shape as number[],
        begin,
        end,
        byteLength: end - begin,
      });
    }
    return new PinnedSafetensorsSource(sourceUrl, descriptors);
  }

  descriptor(name: string): CachedTensorDescriptor {
    const descriptor = this.descriptors.get(name);
    if (!descriptor) throw new Error(`Tensor ${name} is absent from the pinned checkpoint`);
    return descriptor;
  }

  async readTensor(name: string): Promise<CachedTensorPayload> {
    const descriptor = this.descriptor(name);
    const bytes = await fetchExactRange(
      this.sourceUrl,
      descriptor.begin,
      descriptor.byteLength,
    );
    return { ...descriptor, bytes, sha256: await sha256(bytes) };
  }

  async readTensors(names: readonly string[]): Promise<Map<string, CachedTensorPayload>> {
    if (new Set(names).size !== names.length) {
      throw new Error("Pinned safetensors batch contains duplicate tensor names");
    }
    const payloads = await Promise.all(names.map((name) => this.readTensor(name)));
    return new Map(payloads.map((payload) => [payload.name, payload]));
  }
}

async function fetchExactRange(
  sourceUrl: string,
  begin: number,
  byteLength: number,
): Promise<Uint8Array> {
  const end = begin + byteLength - 1;
  const response = await fetch(sourceUrl, {
    headers: { Range: `bytes=${begin}-${end}` },
  });
  if (response.status !== 206 ||
      response.headers.get("content-range") !== `bytes ${begin}-${end}/${MODEL_FILE_SIZE}`) {
    throw new Error(
      `Pinned safetensors range ${begin}-${end} returned ${response.status} ` +
      `${response.headers.get("content-range") ?? "without Content-Range"}`,
    );
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== byteLength) {
    throw new Error(`Pinned safetensors range length mismatch: ${bytes.byteLength} != ${byteLength}`);
  }
  return bytes;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const owned = Uint8Array.from(bytes);
  const digest = await crypto.subtle.digest(
    "SHA-256",
    owned.buffer,
  );
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}