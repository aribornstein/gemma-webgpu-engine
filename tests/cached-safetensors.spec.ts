import { expect, test } from "@playwright/test";

test("initializes and resumes a safetensors cache from byte ranges", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const databaseName = `initialized-safetensors-${crypto.randomUUID()}`;
    const sourceKey = "https://example.invalid/model.safetensors";
    const largeLength = 1024 * 1024 + 1;
    const header = {
      first: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
      second: { dtype: "U8", shape: [3], data_offsets: [4, 7] },
      large: { dtype: "U8", shape: [largeLength], data_offsets: [7, 7 + largeLength] },
    };
    const encoder = new TextEncoder();
    const encodedHeader = encoder.encode(JSON.stringify(header));
    const headerLength = Math.ceil(encodedHeader.byteLength / 8) * 8;
    const dataStart = 8 + headerLength;
    const file = new Uint8Array(dataStart + 7 + largeLength);
    new DataView(file.buffer).setBigUint64(0, BigInt(headerLength), true);
    file.fill(0x20, 8, dataStart);
    file.set(encodedHeader, 8);
    file.set([0, 0, 128, 63, 5, 6, 7], dataStart);
    file[dataStart + 7] = 11;
    file[file.byteLength - 1] = 12;
    const spec = {
      databaseName,
      sourceKey,
      fileSize: file.byteLength,
      dataStart,
      tensorCount: 3,
    };
    let fetchCount = 0;
    let localAvailable = true;
    const localUrl = "/models/test/model.safetensors";
    const signedUrl = "https://cdn.example.invalid/signed-model";
    const requestedRanges: Array<{ url: string; begin: number; end: number }> = [];
    const fetchRange = (async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCount++;
      const url = String(input);
      if (init?.method === "HEAD") {
        return localAvailable
          ? new Response(null, {
              status: 200,
              headers: {
                "content-length": String(file.byteLength),
                "content-type": "application/octet-stream",
              },
            })
          : new Response("<!doctype html>", {
              status: 200,
              headers: { "content-type": "text/html" },
            });
      }
      const range = new Headers(init?.headers).get("range")?.match(/^bytes=(\d+)-(\d+)$/);
      if (!range) return new Response(null, { status: 400 });
      const begin = Number(range[1]);
      const end = Number(range[2]);
      requestedRanges.push({ url, begin, end });
      const response = new Response(file.slice(begin, end + 1), {
        status: 206,
        headers: { "content-range": `bytes ${begin}-${end}/${file.byteLength}` },
      });
      if (url === sourceKey) {
        Object.defineProperty(response, "url", { value: signedUrl });
      }
      return response;
    }) as typeof fetch;

    const initializerPath = "/src/model/safetensors-cache-initializer.ts";
    const cachePath = "/src/model/cached-safetensors.ts";
    const { initializeGemmaSafetensorsCache } = await import(initializerPath);
    const { ReadonlySafetensorsCache } = await import(cachePath);
    const progress: Array<Record<string, unknown>> = [];
    await initializeGemmaSafetensorsCache(
      (event: Record<string, unknown>) => progress.push(event),
      {
        fetch: fetchRange,
        spec,
        concurrency: 2,
        localUrl,
        downloadUrl: sourceKey,
        maximumStandaloneBlobBytes: 1024 * 1024,
      },
    );
    const firstFetchCount = fetchCount;
    const firstRangeRequestCount = requestedRanges.length;
    const cache = await ReadonlySafetensorsCache.open(spec);
    const first = await cache.readTensor("first");
    const second = await cache.readTensor("second");
    const largeEdges = [
      ...(await cache.readTensorSlice("large", 0, 1)),
      ...(await cache.readTensorSlice("large", largeLength - 1, 1)),
    ];
    cache.close();

    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(databaseName);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction("chunks", "readwrite");
        transaction.objectStore("chunks").delete([sourceKey, dataStart + 4, dataStart + 7]);
        transaction.oncomplete = () => {
          database.close();
          resolve();
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });

    localAvailable = false;
    const resumedProgress: Array<Record<string, unknown>> = [];
    await initializeGemmaSafetensorsCache(
      (event: Record<string, unknown>) => resumedProgress.push(event),
      {
        fetch: fetchRange,
        spec,
        concurrency: 2,
        localUrl,
        downloadUrl: sourceKey,
        maximumStandaloneBlobBytes: 1024 * 1024,
      },
    );
    const resumedFetchCount = fetchCount - firstFetchCount;
    const resumedCacheProgress = resumedProgress.find((event) => event.fromCache === true);
    const request = indexedDB.open(databaseName);
    const cacheState = await new Promise<{
      counts: { chunks: number; meta: number };
      cacheFormat: string;
      complete: boolean;
      valueTypes: string[];
    }>((resolve, reject) => {
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction(["chunks", "meta"], "readonly");
        const chunks = transaction.objectStore("chunks").count();
        const meta = transaction.objectStore("meta").count();
        const metadata = transaction.objectStore("meta").get(sourceKey);
        const values = transaction.objectStore("chunks").getAll();
        transaction.oncomplete = () => {
          database.close();
          resolve({
            counts: { chunks: chunks.result, meta: meta.result },
            cacheFormat: metadata.result.cacheFormat,
            complete: metadata.result.complete,
            valueTypes: values.result.map((value) => value.constructor.name),
          });
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });
    await new Promise<void>((resolve, reject) => {
      const deleteRequest = indexedDB.deleteDatabase(databaseName);
      deleteRequest.onsuccess = () => resolve();
      deleteRequest.onerror = () => reject(deleteRequest.error);
    });
    return {
      first: Array.from(first.bytes),
      second: Array.from(second.bytes),
      largeEdges,
      firstFetchCount,
      resumedFetchCount,
      ...cacheState,
      finalProgress: progress.at(-1),
      resumedLoaded: resumedCacheProgress?.loaded,
      resumedTotal: resumedCacheProgress?.total,
      resumedFromCache: resumedCacheProgress?.fromCache,
      initialRangeSources: requestedRanges
        .slice(0, firstRangeRequestCount)
        .map(({ url }) => url),
      resumedRangeSources: requestedRanges
        .slice(firstRangeRequestCount)
        .map(({ url }) => url),
      groupedDownloadSizes: [
        requestedRanges[firstRangeRequestCount - 1]!,
        requestedRanges.at(-1)!,
      ].map(({ begin, end }) => end - begin + 1),
    };
  });

  expect(result).toEqual({
    first: [0, 0, 128, 63],
    second: [5, 6, 7],
    largeEdges: [11, 12],
    firstFetchCount: 4,
    resumedFetchCount: 4,
    counts: { chunks: 4, meta: 1 },
    cacheFormat: "gemma-webgpu-engine-v5-segmented-buffer",
    complete: true,
    valueTypes: ["Blob", "Blob", "Object", "ArrayBuffer"],
    finalProgress: {
      status: "ready",
      kind: "tensors",
      fraction: 1,
      loaded: 3,
      total: 3,
    },
    resumedLoaded: result.resumedLoaded,
    resumedTotal: result.resumedTotal,
    resumedFromCache: true,
    initialRangeSources: [
      "/models/test/model.safetensors",
      "/models/test/model.safetensors",
      "/models/test/model.safetensors",
    ],
    resumedRangeSources: [
      "https://example.invalid/model.safetensors",
      "https://example.invalid/model.safetensors",
      "https://example.invalid/model.safetensors",
    ],
    groupedDownloadSizes: [1024 * 1024 + 8, 1024 * 1024 + 8],
  });
  expect(result.resumedLoaded).toBeLessThan(result.resumedTotal as number);
});

test("reads pinned safetensors tensors without mutating the existing cache", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const databaseName = `readonly-safetensors-${crypto.randomUUID()}`;
    const sourceKey = "https://example.invalid/pinned/model.safetensors";
    const dataStart = 64;
    const firstBytes = new Uint8Array([0, 0, 128, 63]);

    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(databaseName, 2);
      request.onupgradeneeded = () => {
        request.result.createObjectStore("chunks");
        request.result.createObjectStore("meta");
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction(["chunks", "meta"], "readwrite");
        transaction.objectStore("meta").put({
          size: 71,
          dataStart,
          header: {
            first: { dtype: "F32", shape: [1], data_offsets: [0, 4] },
            second: { dtype: "U8", shape: [3], data_offsets: [4, 7] },
          },
        }, sourceKey);
        transaction.objectStore("chunks").put(
          new Blob([firstBytes]),
          [sourceKey, dataStart, dataStart + firstBytes.byteLength],
        );
        transaction.oncomplete = () => {
          database.close();
          resolve();
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });

    const modulePath = "/src/model/cached-safetensors.ts";
    const { ReadonlySafetensorsCache } = await import(modulePath);
    const spec = { databaseName, sourceKey, fileSize: 71, dataStart, tensorCount: 2 };
    const cache = await ReadonlySafetensorsCache.open(spec);
    let missingError = "";
    try {
      await cache.readTensor("second");
    } catch (error) {
      missingError = error instanceof Error ? error.message : String(error);
    }
    const payload = await cache.readTensor("first");
    const slice = await cache.readTensorSlice("first", 1, 2);
    const slices = await cache.readTensorSlices([
      { name: "first", byteOffset: 0, byteLength: 1 },
      { name: "first", byteOffset: 2, byteLength: 2 },
    ]);
    const batchPayload = (await cache.readTensors(["first"])).get("first");
    const tensorNames = cache.tensorNames();
    const descriptor = cache.descriptor("second");
    cache.close();

    let identityError = "";
    try {
      await ReadonlySafetensorsCache.open({ ...spec, fileSize: 72 });
    } catch (error) {
      identityError = error instanceof Error ? error.message : String(error);
    }

    const counts = await new Promise<{ chunks: number; meta: number }>((resolve, reject) => {
      const request = indexedDB.open(databaseName, 2);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction(["chunks", "meta"], "readonly");
        const chunks = transaction.objectStore("chunks").count();
        const meta = transaction.objectStore("meta").count();
        transaction.oncomplete = () => {
          database.close();
          resolve({ chunks: chunks.result, meta: meta.result });
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });

    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.deleteDatabase(databaseName);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    return {
      bytes: Array.from(payload.bytes),
      slice: Array.from(slice),
      slices: slices.map((value: Uint8Array) => Array.from(value)),
      sha256: payload.sha256,
      batchSha256: batchPayload?.sha256,
      tensorNames,
      descriptor,
      missingError,
      identityError,
      counts,
    };
  });

  expect(result.bytes).toEqual([0, 0, 128, 63]);
  expect(result.slice).toEqual([0, 128]);
  expect(result.slices).toEqual([[0], [128, 63]]);
  expect(result.sha256).toBe("e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c");
  expect(result.batchSha256).toBe(result.sha256);
  expect(result.tensorNames).toEqual(["first", "second"]);
  expect(result.descriptor).toEqual({
    name: "second",
    dtype: "U8",
    shape: [3],
    begin: 68,
    end: 71,
    byteLength: 3,
  });
  expect(result.missingError).toContain("not present in the readonly cache");
  expect(result.identityError).toContain("does not match the pinned manifest");
  expect(result.counts).toEqual({ chunks: 1, meta: 1 });
});