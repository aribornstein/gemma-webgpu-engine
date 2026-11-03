import { expect, test } from "@playwright/test";

test("rejects a download whose header differs from the pinned digest", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const header = new TextEncoder().encode(JSON.stringify({
      only: { dtype: "U8", shape: [4], data_offsets: [0, 4] },
    }));
    const headerLength = Math.ceil(header.byteLength / 8) * 8;
    const dataStart = 8 + headerLength;
    const file = new Uint8Array(dataStart + 4);
    new DataView(file.buffer).setBigUint64(0, BigInt(headerLength), true);
    file.fill(0x20, 8, dataStart);
    file.set(header, 8);
    file.set([1, 2, 3, 4], dataStart);
    const digest = new Uint8Array(
      await crypto.subtle.digest("SHA-256", file.slice(0, dataStart)),
    );
    const headerSha256 = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
    const fetchRange = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      const range = new Headers(init?.headers).get("range")?.match(/^bytes=(\d+)-(\d+)$/);
      if (!range) return new Response(null, { status: 400 });
      const begin = Number(range[1]);
      const end = Number(range[2]);
      return new Response(file.slice(begin, end + 1), {
        status: 206,
        headers: { "content-range": `bytes ${begin}-${end}/${file.byteLength}` },
      });
    }) as typeof fetch;
    const initializerPath = "/src/model/safetensors-cache-initializer.ts";
    const { initializeGemmaSafetensorsCache } = await import(initializerPath);
    const run = async (expectedSha256: string): Promise<string> => {
      const spec = {
        databaseName: `integrity-${crypto.randomUUID()}`,
        sourceKey: "https://example.invalid/model.safetensors",
        fileSize: file.byteLength,
        dataStart,
        tensorCount: 1,
        headerSha256: expectedSha256,
      };
      try {
        await initializeGemmaSafetensorsCache(() => undefined, {
          fetch: fetchRange,
          spec,
          localUrl: null,
          downloadUrl: spec.sourceKey,
        });
        return "ok";
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      } finally {
        indexedDB.deleteDatabase(spec.databaseName);
      }
    };
    return {
      matching: await run(headerSha256),
      mismatching: await run("0".repeat(64)),
    };
  });
  expect(result.matching).toBe("ok");
  expect(result.mismatching).toContain("does not match the pinned manifest");
});

test("refuses to open a cache whose download is marked incomplete", async ({ page }) => {
  await page.goto("/");
  const message = await page.evaluate(async () => {
    const databaseName = `incomplete-${crypto.randomUUID()}`;
    const sourceKey = "https://example.invalid/model.safetensors";
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(databaseName, 2);
      request.onupgradeneeded = () => {
        request.result.createObjectStore("chunks");
        request.result.createObjectStore("meta");
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction("meta", "readwrite");
        transaction.objectStore("meta").put({
          size: 20,
          dataStart: 16,
          header: { only: { dtype: "U8", shape: [4], data_offsets: [0, 4] } },
          complete: false,
        }, sourceKey);
        transaction.oncomplete = () => {
          database.close();
          resolve();
        };
        transaction.onerror = () => reject(transaction.error);
      };
    });
    const modulePath = "/src/model/cached-safetensors.ts";
    const { ReadonlySafetensorsCache } = await import(modulePath);
    try {
      const cache = await ReadonlySafetensorsCache.open({
        databaseName,
        sourceKey,
        fileSize: 20,
        dataStart: 16,
        tensorCount: 1,
      });
      cache.close();
      return "opened";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    } finally {
      indexedDB.deleteDatabase(databaseName);
    }
  });
  expect(message).toContain("download is incomplete");
});
