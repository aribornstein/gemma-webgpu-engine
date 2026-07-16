import { expect, test } from "@playwright/test";

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