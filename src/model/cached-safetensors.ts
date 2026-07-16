const CHUNKS_STORE = "chunks";
const META_STORE = "meta";

export interface SafetensorsCacheSpec {
  databaseName: string;
  sourceKey: string;
  fileSize: number;
  dataStart: number;
  tensorCount: number;
}

export interface CachedTensorDescriptor {
  name: string;
  dtype: string;
  shape: readonly number[];
  begin: number;
  end: number;
  byteLength: number;
}

export interface CachedTensorPayload extends CachedTensorDescriptor {
  bytes: Uint8Array;
  sha256: string;
}

export interface CachedTensorSliceRequest {
  name: string;
  byteOffset: number;
  byteLength: number;
}

export const GEMMA_4_E2B_CACHE_SPEC: Readonly<SafetensorsCacheSpec> = {
  databaseName: "safetensors-cache-v1",
  sourceKey:
    "https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors",
  fileSize: 2_458_111_846,
  dataStart: 375_400,
  tensorCount: 2_780,
};

export class ReadonlySafetensorsCache {
  readonly spec: Readonly<SafetensorsCacheSpec>;
  readonly descriptors: ReadonlyMap<string, CachedTensorDescriptor>;
  private readonly database: IDBDatabase;
  private readonly blobValues = new Map<string, Blob>();

  private constructor(
    database: IDBDatabase,
    spec: Readonly<SafetensorsCacheSpec>,
    descriptors: Map<string, CachedTensorDescriptor>,
  ) {
    this.database = database;
    this.spec = spec;
    this.descriptors = descriptors;
  }

  static async open(
    spec: Readonly<SafetensorsCacheSpec> = GEMMA_4_E2B_CACHE_SPEC,
  ): Promise<ReadonlySafetensorsCache> {
    if (typeof indexedDB === "undefined" || typeof indexedDB.databases !== "function") {
      throw new Error("IndexedDB database enumeration is unavailable");
    }
    const databaseInfo = (await indexedDB.databases()).find(
      (candidate) => candidate.name === spec.databaseName,
    );
    if (!databaseInfo || databaseInfo.version === undefined) {
      throw new Error(`IndexedDB ${spec.databaseName} does not exist on this origin`);
    }

    const database = await openExistingDatabase(spec.databaseName, databaseInfo.version);
    try {
      if (
        !database.objectStoreNames.contains(CHUNKS_STORE) ||
        !database.objectStoreNames.contains(META_STORE)
      ) {
        throw new Error(`IndexedDB ${spec.databaseName} does not match the safetensors schema`);
      }
      const metadata = await readStoreValue(database, META_STORE, spec.sourceKey);
      const descriptors = parseMetadata(metadata, spec);
      return new ReadonlySafetensorsCache(database, spec, descriptors);
    } catch (error) {
      database.close();
      throw error;
    }
  }

  tensorNames(prefix = ""): string[] {
    return Array.from(this.descriptors.keys()).filter((name) => name.startsWith(prefix));
  }

  descriptor(name: string): CachedTensorDescriptor {
    const descriptor = this.descriptors.get(name);
    if (!descriptor) throw new Error(`Tensor ${name} is absent from cached metadata`);
    return descriptor;
  }

  async readTensor(name: string): Promise<CachedTensorPayload> {
    const tensors = await this.readTensors([name]);
    const tensor = tensors.get(name);
    if (!tensor) throw new Error(`Tensor ${name} batch read did not return a payload`);
    return tensor;
  }

  async readTensorSlice(
    name: string,
    byteOffset: number,
    byteLength: number,
  ): Promise<Uint8Array> {
    const descriptor = this.descriptor(name);
    if (!Number.isInteger(byteOffset) || byteOffset < 0 ||
        !Number.isInteger(byteLength) || byteLength < 1 ||
        byteOffset + byteLength > descriptor.byteLength) {
      throw new Error(`Tensor ${name} slice exceeds its ${descriptor.byteLength}-byte range`);
    }
    const value = this.blobValues.get(name) ?? await readStoreValue(
      this.database,
      CHUNKS_STORE,
      [this.spec.sourceKey, descriptor.begin, descriptor.end],
    );
    if (value instanceof Blob) this.blobValues.set(name, value);
    const bytes = await copyBinarySlice(value, byteOffset, byteLength);
    if (!bytes) {
      throw new Error(`Tensor ${name} is not present in the readonly cache`);
    }
    return bytes;
  }

  async readTensorSlices(
    requests: readonly CachedTensorSliceRequest[],
  ): Promise<Uint8Array[]> {
    const descriptors = requests.map(({ name, byteOffset, byteLength }) => {
      const descriptor = this.descriptor(name);
      if (!Number.isInteger(byteOffset) || byteOffset < 0 ||
          !Number.isInteger(byteLength) || byteLength < 1 ||
          byteOffset + byteLength > descriptor.byteLength) {
        throw new Error(`Tensor ${name} slice exceeds its ${descriptor.byteLength}-byte range`);
      }
      return descriptor;
    });
    const uniqueDescriptors = Array.from(
      new Map(descriptors.map((descriptor) => [descriptor.name, descriptor])).values(),
    );
    const missingDescriptors = uniqueDescriptors.filter(
      ({ name }) => !this.blobValues.has(name),
    );
    const missingValues = await readStoreValues(
      this.database,
      CHUNKS_STORE,
      missingDescriptors.map(({ begin, end }) => [this.spec.sourceKey, begin, end]),
    );
    const valuesByName = new Map<string, unknown>();
    for (const descriptor of uniqueDescriptors) {
      const value = this.blobValues.get(descriptor.name);
      if (value) valuesByName.set(descriptor.name, value);
    }
    missingDescriptors.forEach((descriptor, index) => {
      const value = missingValues[index];
      valuesByName.set(descriptor.name, value);
      if (value instanceof Blob) this.blobValues.set(descriptor.name, value);
    });
    return Promise.all(requests.map(async ({ name, byteOffset, byteLength }) => {
      const bytes = await copyBinarySlice(valuesByName.get(name), byteOffset, byteLength);
      if (!bytes) throw new Error(`Tensor ${name} is not present in the readonly cache`);
      return bytes;
    }));
  }

  async readTensors(names: readonly string[]): Promise<Map<string, CachedTensorPayload>> {
    if (new Set(names).size !== names.length) {
      throw new Error("Readonly safetensors batch contains duplicate tensor names");
    }
    const descriptors = names.map((name) => this.descriptor(name));
    const values = await readStoreValues(
      this.database,
      CHUNKS_STORE,
      descriptors.map(({ begin, end }) => [this.spec.sourceKey, begin, end]),
    );
    const payloads = await Promise.all(descriptors.map(async (descriptor, index) => {
      const bytes = await copyBinary(values[index]);
      if (!bytes) {
        throw new Error(`Tensor ${descriptor.name} is not present in the readonly cache`);
      }
      if (bytes.byteLength !== descriptor.byteLength) {
        throw new Error(
          `Tensor ${descriptor.name} cache length mismatch: ` +
          `${bytes.byteLength} != ${descriptor.byteLength}`,
        );
      }
      return { ...descriptor, bytes, sha256: await sha256(bytes) };
    }));
    return new Map(payloads.map((payload) => [payload.name, payload]));
  }

  close(): void {
    this.blobValues.clear();
    this.database.close();
  }
}

function openExistingDatabase(name: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    let upgradeAttempted = false;
    request.onupgradeneeded = () => {
      upgradeAttempted = true;
      request.transaction?.abort();
    };
    request.onsuccess = () => {
      if (upgradeAttempted) {
        request.result.close();
        reject(new Error(`Refused to create or upgrade IndexedDB ${name}`));
        return;
      }
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onerror = () => reject(
      upgradeAttempted
        ? new Error(`Refused to create or upgrade IndexedDB ${name}`)
        : (request.error ?? new Error(`Failed to open IndexedDB ${name}`)),
    );
    request.onblocked = () => reject(new Error(`IndexedDB ${name} open was blocked`));
  });
}

function readStoreValue(
  database: IDBDatabase,
  storeName: string,
  key: IDBValidKey,
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(storeName, "readonly");
    const request = transaction.objectStore(storeName).get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error(`${storeName} read aborted`));
  });
}

function readStoreValues(
  database: IDBDatabase,
  storeName: string,
  keys: readonly IDBValidKey[],
): Promise<unknown[]> {
  return new Promise((resolve, reject) => {
    if (keys.length === 0) {
      resolve([]);
      return;
    }
    const transaction = database.transaction(storeName, "readonly");
    const store = transaction.objectStore(storeName);
    const values: unknown[] = new Array(keys.length);
    keys.forEach((key, index) => {
      const request = store.get(key);
      request.onsuccess = () => {
        values[index] = request.result;
      };
      request.onerror = () => reject(request.error);
    });
    transaction.oncomplete = () => resolve(values);
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(
      transaction.error ?? new Error(`${storeName} batch read aborted`),
    );
  });
}

function parseMetadata(
  value: unknown,
  spec: Readonly<SafetensorsCacheSpec>,
): Map<string, CachedTensorDescriptor> {
  if (!isRecord(value)) throw new Error(`Cached metadata is missing for ${spec.sourceKey}`);
  if (value.size !== spec.fileSize || value.dataStart !== spec.dataStart) {
    throw new Error("Cached safetensors file identity does not match the pinned manifest");
  }
  if (!isRecord(value.header)) throw new Error("Cached safetensors header is missing");

  const entries = Object.entries(value.header).filter(([name]) => name !== "__metadata__");
  if (entries.length !== spec.tensorCount) {
    throw new Error(`Cached safetensors tensor count mismatch: ${entries.length}`);
  }

  const descriptors = new Map<string, CachedTensorDescriptor>();
  for (const [name, rawDescriptor] of entries) {
    if (!isRecord(rawDescriptor) || typeof rawDescriptor.dtype !== "string") {
      throw new Error(`Cached safetensors descriptor is invalid for ${name}`);
    }
    const shape = numberArray(rawDescriptor.shape);
    const offsets = numberPair(rawDescriptor.data_offsets);
    if (!shape || !offsets || offsets[0] < 0 || offsets[1] < offsets[0]) {
      throw new Error(`Cached safetensors descriptor is invalid for ${name}`);
    }
    const begin = spec.dataStart + offsets[0];
    const end = spec.dataStart + offsets[1];
    if (end > spec.fileSize) {
      throw new Error(`Cached safetensors tensor range exceeds the pinned file for ${name}`);
    }
    descriptors.set(name, {
      name,
      dtype: rawDescriptor.dtype,
      shape,
      begin,
      end,
      byteLength: end - begin,
    });
  }
  return descriptors;
}

async function copyBinary(value: unknown): Promise<Uint8Array | null> {
  let source: Uint8Array;
  if (value instanceof Blob) {
    source = new Uint8Array(await value.arrayBuffer());
  } else if (value instanceof ArrayBuffer) {
    source = new Uint8Array(value);
  } else if (ArrayBuffer.isView(value)) {
    source = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  } else {
    return null;
  }
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  return copy;
}

async function copyBinarySlice(
  value: unknown,
  byteOffset: number,
  byteLength: number,
): Promise<Uint8Array | null> {
  if (value instanceof Blob) {
    return new Uint8Array(await value.slice(byteOffset, byteOffset + byteLength).arrayBuffer());
  }
  let source: Uint8Array;
  if (value instanceof ArrayBuffer) {
    source = new Uint8Array(value);
  } else if (ArrayBuffer.isView(value)) {
    source = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  } else {
    return null;
  }
  const output = new Uint8Array(byteLength);
  output.set(source.subarray(byteOffset, byteOffset + byteLength));
  return output;
}

function numberPair(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const first = nonnegativeInteger(value[0]);
  const second = nonnegativeInteger(value[1]);
  return first === null || second === null ? null : [first, second];
}

function numberArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  const numbers = value.map(nonnegativeInteger);
  return numbers.some((dimension) => dimension === null) ? null : numbers as number[];
}

function nonnegativeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}