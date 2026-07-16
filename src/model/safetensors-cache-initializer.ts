import {
  GEMMA_4_E2B_CACHE_SPEC,
  type SafetensorsCacheSpec,
} from "./cached-safetensors";
import {
  GEMMA_LOCAL_SAFETENSORS_URL,
  GEMMA_SAFETENSORS_DOWNLOAD_URL,
} from "./pinned-safetensors";

const CHUNKS_STORE = "chunks";
const META_STORE = "meta";
const CACHE_VERSION = 2;
const DEFAULT_CONCURRENCY = 1;
const CACHE_FORMAT = "gemma-webgpu-engine-v5-segmented-buffer";
const OWNED_BLOB_CACHE_FORMAT = "gemma-webgpu-engine-v2-owned-blob";
const SEGMENTED_BLOB_CACHE_FORMAT = "gemma-webgpu-engine-v3-segmented-blob";
const LARGE_BLOB_CACHE_FORMAT = "gemma-webgpu-engine-v4-segmented-buffer";
const MAX_GROUP_BYTES = 128 * 1024 * 1024;
const MAX_GROUP_GAP = 1024 * 1024;
const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
const RANGE_FETCH_ATTEMPTS = 3;

export interface SafetensorsCacheInitializationProgress {
  status?: "weights" | "ready" | string;
  kind?: "bytes" | "tensors" | string;
  fraction?: number;
  loaded?: number;
  total?: number;
  fromCache?: boolean;
  message?: string;
}

export interface SafetensorsCacheInitializationOptions {
  fetch?: typeof fetch;
  spec?: Readonly<SafetensorsCacheSpec>;
  concurrency?: number;
  downloadUrl?: string;
  localUrl?: string | null;
  maximumStandaloneBlobBytes?: number;
}

interface TensorRange {
  begin: number;
  end: number;
}

interface TensorRangeGroup extends TensorRange {
  ranges: TensorRange[];
}

interface RangeSource {
  url: string;
}

interface CacheMetadata {
  size: number;
  dataStart: number;
  acceptsRanges: true;
  header: Record<string, unknown>;
  cacheFormat: typeof CACHE_FORMAT;
  complete: boolean;
}

export async function initializeGemmaSafetensorsCache(
  onProgress: (progress: SafetensorsCacheInitializationProgress) => void,
  options: SafetensorsCacheInitializationOptions = {},
): Promise<void> {
  const spec = options.spec ?? GEMMA_4_E2B_CACHE_SPEC;
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const remoteUrl = options.downloadUrl ?? GEMMA_SAFETENSORS_DOWNLOAD_URL;
  const localUrl = options.localUrl === undefined ? GEMMA_LOCAL_SAFETENSORS_URL : options.localUrl;
  const concurrency = options.concurrency ?? DEFAULT_CONCURRENCY;
  const maximumStandaloneBlobBytes = options.maximumStandaloneBlobBytes ?? MAX_REQUEST_BYTES;
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 8) {
    throw new Error("Safetensors download concurrency must be between 1 and 8");
  }
  if (!Number.isInteger(maximumStandaloneBlobBytes) || maximumStandaloneBlobBytes < 1) {
    throw new Error("Safetensors standalone Blob limit must be a positive integer");
  }

  await navigator.storage?.persist?.().catch(() => false);
  const downloadUrl = await resolveDownloadUrl(fetcher, localUrl, remoteUrl, spec, onProgress);
  const source = { url: downloadUrl };
  const metadata = await fetchMetadata(fetcher, source, spec);
  const ranges = tensorRanges(metadata, spec);
  const database = await openWritableDatabase(spec.databaseName);
  try {
    const existingMetadata = await readValue(database, META_STORE, spec.sourceKey);
    const canResume = metadataMatches(existingMetadata, metadata);
    if (!canResume) {
      await deleteSourceChunks(database);
      await writeValue(database, META_STORE, metadata, spec.sourceKey);
    }

    let existingRanges = new Set<string>();
    if (canResume) {
      try {
        existingRanges = await readExistingRanges(
          database,
          spec.sourceKey,
          isRecord(existingMetadata) &&
              (existingMetadata.cacheFormat === OWNED_BLOB_CACHE_FORMAT ||
                existingMetadata.cacheFormat === SEGMENTED_BLOB_CACHE_FORMAT ||
                existingMetadata.cacheFormat === LARGE_BLOB_CACHE_FORMAT)
            ? maximumStandaloneBlobBytes
            : Number.POSITIVE_INFINITY,
        );
      } catch (error) {
        if (!isNotReadableError(error)) throw error;
        await deleteSourceChunks(database);
      }
    }
    let loaded = spec.dataStart + ranges.reduce(
      (total, range) => total + (existingRanges.has(rangeKey(range)) ? range.end - range.begin : 0),
      0,
    );
    let reportedLoaded = loaded;
    const reportLoaded = (candidate: number): void => {
      reportedLoaded = Math.max(reportedLoaded, Math.min(candidate, spec.fileSize));
      reportBytes(onProgress, reportedLoaded, spec.fileSize, false);
    };
    reportBytes(onProgress, loaded, spec.fileSize, existingRanges.size > 0);

    for (const range of ranges) {
      if (range.begin !== range.end || existingRanges.has(rangeKey(range))) continue;
      await writeValue(
        database,
        CHUNKS_STORE,
        new Blob([new ArrayBuffer(0)]),
        [spec.sourceKey, range.begin, range.end],
      );
    }

    const groups = groupTensorRanges(ranges.filter((range) => range.begin !== range.end));
    let nextIndex = 0;
    const downloadNext = async (): Promise<void> => {
      while (nextIndex < groups.length) {
        const group = groups[nextIndex++];
        const missingRanges = group.ranges.filter((range) => !existingRanges.has(rangeKey(range)));
        if (missingRanges.length === 0) continue;
        const value = await fetchGroupedRange(
          fetcher,
          source,
          group,
          spec.fileSize,
          (fetchedThrough) => {
            const fetchedTensorBytes = missingRanges.reduce(
              (total, range) => total + Math.max(
                0,
                Math.min(range.end, fetchedThrough) - range.begin,
              ),
              0,
            );
            reportLoaded(loaded + fetchedTensorBytes);
          },
        );
        if (value.byteLength !== group.end - group.begin) {
          throw new Error(
            `Safetensors range ${group.begin}-${group.end - 1} returned ${value.byteLength} bytes`,
          );
        }
        for (const range of missingRanges) {
          const tensor = value.slice(
            range.begin - group.begin,
            range.end - group.begin,
          ).buffer;
          await writeTensorValue(
            database,
            spec.sourceKey,
            range,
            tensor,
            maximumStandaloneBlobBytes,
          );
          loaded += tensor.byteLength;
          reportLoaded(loaded);
        }
      }
    };
    await Promise.all(Array.from(
      { length: Math.min(concurrency, groups.length) },
      () => downloadNext(),
    ));
    await writeValue(
      database,
      META_STORE,
      { ...metadata, complete: true },
      spec.sourceKey,
    );
    onProgress({
      status: "ready",
      kind: "tensors",
      fraction: 1,
      loaded: ranges.length,
      total: ranges.length,
    });
  } finally {
    database.close();
  }
}

async function resolveDownloadUrl(
  fetcher: typeof fetch,
  localUrl: string | null,
  remoteUrl: string,
  spec: Readonly<SafetensorsCacheSpec>,
  onProgress: (progress: SafetensorsCacheInitializationProgress) => void,
): Promise<string> {
  if (localUrl !== null) {
    const localAvailable = await probeSameOriginSource(fetcher, localUrl, spec, "Local model");
    if (localAvailable) {
      onProgress({
        status: "weights",
        kind: "bytes",
        fraction: 0,
        message: "Reading model weights from public/models",
      });
      return localUrl;
    }
  }
  onProgress({
    status: "weights",
    kind: "bytes",
    fraction: 0,
    message: "Local model unavailable; connecting directly to Hugging Face",
  });
  return remoteUrl;
}

async function fetchGroupedRange(
  fetcher: typeof fetch,
  source: RangeSource,
  group: TensorRangeGroup,
  fileSize: number,
  onFetchedThrough: (endExclusive: number) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  const value = new Uint8Array(group.end - group.begin);
  for (let begin = group.begin; begin < group.end; begin += MAX_REQUEST_BYTES) {
    const inclusiveEnd = Math.min(group.end, begin + MAX_REQUEST_BYTES) - 1;
    value.set(
      await fetchRange(fetcher, source, begin, inclusiveEnd, fileSize),
      begin - group.begin,
    );
    onFetchedThrough(inclusiveEnd + 1);
  }
  return value;
}

async function probeSameOriginSource(
  fetcher: typeof fetch,
  url: string,
  spec: Readonly<SafetensorsCacheSpec>,
  label: string,
): Promise<boolean> {
  let response: Response;
  try {
    response = await fetcher(url, { method: "HEAD" });
  } catch {
    return false;
  }
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  const missing = response.status === 404 ||
    (response.ok && contentType.includes("text/html"));
  if (missing) return false;
  if (!response.ok) throw new Error(`${label} check failed with HTTP ${response.status}`);
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null && Number(contentLength) !== spec.fileSize) {
    throw new Error(
      `${label} size does not match the pinned manifest: ${contentLength} != ${spec.fileSize}`,
    );
  }
  return true;
}

async function fetchMetadata(
  fetcher: typeof fetch,
  source: RangeSource,
  spec: Readonly<SafetensorsCacheSpec>,
): Promise<CacheMetadata> {
  const prefix = await fetchRange(fetcher, source, 0, 7, spec.fileSize);
  const headerLength = Number(new DataView(prefix.buffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2) {
    throw new Error("Safetensors header length is invalid");
  }
  const dataStart = 8 + headerLength;
  if (dataStart !== spec.dataStart) {
    throw new Error(
      `Safetensors data offset does not match the pinned manifest: ${dataStart}`,
    );
  }
  const headerBytes = (await fetchRange(
    fetcher,
    source,
    0,
    dataStart - 1,
    spec.fileSize,
  )).subarray(8);
  const parsed: unknown = JSON.parse(new TextDecoder().decode(headerBytes));
  if (!isRecord(parsed)) throw new Error("Safetensors header is not an object");
  const tensorCount = Object.keys(parsed).filter((name) => name !== "__metadata__").length;
  if (tensorCount !== spec.tensorCount) {
    throw new Error(`Safetensors tensor count does not match the pinned manifest: ${tensorCount}`);
  }
  return {
    size: spec.fileSize,
    dataStart,
    acceptsRanges: true,
    header: parsed,
    cacheFormat: CACHE_FORMAT,
    complete: false,
  };
}

function tensorRanges(
  metadata: CacheMetadata,
  spec: Readonly<SafetensorsCacheSpec>,
): TensorRange[] {
  const ranges = Object.entries(metadata.header)
    .filter(([name]) => name !== "__metadata__")
    .map(([name, value]) => {
      if (!isRecord(value) || !Array.isArray(value.data_offsets) ||
          value.data_offsets.length !== 2) {
        throw new Error(`Safetensors descriptor is invalid for ${name}`);
      }
      const [relativeBegin, relativeEnd] = value.data_offsets;
      if (!Number.isSafeInteger(relativeBegin) || !Number.isSafeInteger(relativeEnd) ||
          (relativeBegin as number) < 0 || (relativeEnd as number) < (relativeBegin as number)) {
        throw new Error(`Safetensors offsets are invalid for ${name}`);
      }
      const begin = metadata.dataStart + (relativeBegin as number);
      const end = metadata.dataStart + (relativeEnd as number);
      if (end > spec.fileSize) throw new Error(`Safetensors range exceeds the file for ${name}`);
      return { begin, end };
    });
  ranges.sort((left, right) => left.begin - right.begin);
  return ranges;
}

function groupTensorRanges(ranges: readonly TensorRange[]): TensorRangeGroup[] {
  const groups: TensorRangeGroup[] = [];
  let current: TensorRangeGroup | null = null;
  for (const range of ranges) {
    if (current !== null) {
      const gap = range.begin - current.end;
      const groupedBytes = range.end - current.begin;
      if (gap <= MAX_GROUP_GAP && groupedBytes <= MAX_GROUP_BYTES) {
        current.end = range.end;
        current.ranges.push(range);
        continue;
      }
      groups.push(current);
    }
    current = { begin: range.begin, end: range.end, ranges: [range] };
  }
  if (current !== null) groups.push(current);
  return groups;
}

async function fetchRange(
  fetcher: typeof fetch,
  source: RangeSource,
  begin: number,
  inclusiveEnd: number,
  fileSize: number,
): Promise<Uint8Array<ArrayBuffer>> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= RANGE_FETCH_ATTEMPTS; attempt++) {
    try {
      const response = await fetcher(source.url, {
        headers: { Range: `bytes=${begin}-${inclusiveEnd}` },
      });
      const expectedRange = `bytes ${begin}-${inclusiveEnd}/${fileSize}`;
      if (response.status !== 206 || response.headers.get("content-range") !== expectedRange) {
        await response.body?.cancel();
        throw new Error(
          `status ${response.status}, ${response.headers.get("content-range") ?? "no content-range"}`,
        );
      }
      return new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      if (attempt < RANGE_FETCH_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, 250 * attempt));
      }
    }
  }
  throw new Error(
    `Failed to fetch model range ${begin}-${inclusiveEnd} after ` +
    `${RANGE_FETCH_ATTEMPTS} attempts: ${errorMessage(lastError)}`,
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function metadataMatches(value: unknown, expected: CacheMetadata): boolean {
  return isRecord(value) && value.size === expected.size && value.dataStart === expected.dataStart &&
    (value.cacheFormat === expected.cacheFormat || value.cacheFormat === OWNED_BLOB_CACHE_FORMAT ||
      value.cacheFormat === SEGMENTED_BLOB_CACHE_FORMAT ||
      value.cacheFormat === LARGE_BLOB_CACHE_FORMAT) &&
    isRecord(value.header) && JSON.stringify(value.header) === JSON.stringify(expected.header);
}

async function openWritableDatabase(name: string): Promise<IDBDatabase> {
  const databaseInfo = (await indexedDB.databases()).find((candidate) => candidate.name === name);
  const version = Math.max(CACHE_VERSION, databaseInfo?.version ?? 0);
  let database = await openDatabase(name, version);
  if (database.objectStoreNames.contains(CHUNKS_STORE) &&
      database.objectStoreNames.contains(META_STORE)) {
    return database;
  }
  database.close();
  database = await openDatabase(name, version + 1);
  return database;
}

function openDatabase(name: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(CHUNKS_STORE)) {
        request.result.createObjectStore(CHUNKS_STORE);
      }
      if (!request.result.objectStoreNames.contains(META_STORE)) {
        request.result.createObjectStore(META_STORE);
      }
    };
    request.onsuccess = () => {
      request.result.onversionchange = () => request.result.close();
      resolve(request.result);
    };
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error(`IndexedDB ${name} upgrade was blocked`));
  });
}

function readValue(database: IDBDatabase, storeName: string, key: IDBValidKey): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(storeName, "readonly");
    const request = transaction.objectStore(storeName).get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    transaction.onerror = () => reject(transaction.error);
  });
}

function writeValue(
  database: IDBDatabase,
  storeName: string,
  value: unknown,
  key: IDBValidKey,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(storeName, "readwrite");
    transaction.objectStore(storeName).put(value, key);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error(`${storeName} write aborted`));
  });
}

async function readExistingRanges(
  database: IDBDatabase,
  sourceKey: string,
  maximumRangeBytes: number,
): Promise<Set<string>> {
  return new Promise<Set<string>>((resolve, reject) => {
    const transaction = database.transaction(CHUNKS_STORE, "readonly");
    const ranges = new Set<string>();
    const request = transaction.objectStore(CHUNKS_STORE).openKeyCursor();
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) return;
      const key = cursor.key;
      if (Array.isArray(key) && key.length === 3 && key[0] === sourceKey &&
          typeof key[1] === "number" && typeof key[2] === "number" &&
          key[2] - key[1] <= maximumRangeBytes) {
        ranges.add(`${key[1]}:${key[2]}`);
      }
      cursor.continue();
    };
    request.onerror = () => reject(request.error);
    transaction.oncomplete = () => resolve(ranges);
    transaction.onerror = () => reject(transaction.error);
  });
}

async function writeTensorValue(
  database: IDBDatabase,
  sourceKey: string,
  range: TensorRange,
  tensor: ArrayBuffer,
  maximumStandaloneBlobBytes: number,
): Promise<void> {
  if (tensor.byteLength <= maximumStandaloneBlobBytes) {
    await writeValue(
      database,
      CHUNKS_STORE,
      new Blob([tensor]),
      [sourceKey, range.begin, range.end],
    );
    return;
  }
  const bytes = new Uint8Array(tensor);
  let partCount = 0;
  for (let begin = 0; begin < bytes.byteLength; begin += MAX_REQUEST_BYTES) {
    await writeValue(
      database,
      CHUNKS_STORE,
      bytes.slice(begin, begin + MAX_REQUEST_BYTES).buffer,
      [sourceKey, range.begin, range.end, partCount],
    );
    partCount++;
  }
  await writeValue(
    database,
    CHUNKS_STORE,
    {
      kind: "segmented-array-buffer-v1",
      byteLength: tensor.byteLength,
      partBytes: MAX_REQUEST_BYTES,
      partCount,
    },
    [sourceKey, range.begin, range.end],
  );
}

function deleteSourceChunks(database: IDBDatabase): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(CHUNKS_STORE, "readwrite");
    const request = transaction.objectStore(CHUNKS_STORE).clear();
    request.onerror = () => reject(request.error);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error("Cache cleanup aborted"));
  });
}

function isNotReadableError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotReadableError";
}

function reportBytes(
  onProgress: (progress: SafetensorsCacheInitializationProgress) => void,
  loaded: number,
  total: number,
  fromCache: boolean,
): void {
  onProgress({
    status: "weights",
    kind: "bytes",
    fraction: loaded / total,
    loaded,
    total,
    fromCache,
  });
}

function rangeKey(range: TensorRange): string {
  return `${range.begin}:${range.end}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}