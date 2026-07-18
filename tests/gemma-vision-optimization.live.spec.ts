import { expect, test } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const LIVE = process.env.GEMMA_VISION_OPTIMIZATION === "1";
const ROWS = 2520;
const SAMPLES_PER_MODE = 10;

test("benchmarks the maximum-budget vision dense tile", async ({ page }) => {
  test.skip(!LIVE, "Set GEMMA_VISION_OPTIMIZATION=1 to run the live vision benchmark");
  test.setTimeout(600_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async ({ rows, samplesPerMode }) => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-vision-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const layerPath = "/src/webgpu/vision-layer.ts";
    const [{ PinnedSafetensorsSource }, { loadGemmaVisionLayerWeights },
      { getWebGpuDevice }, layer] = await Promise.all([
      import(sourcePath),
      import(weightsPath),
      import(devicePath),
      import(layerPath),
    ]);
    const source = await PinnedSafetensorsSource.open();
    const weights = await loadGemmaVisionLayerWeights(source, 0);
    const device = await getWebGpuDevice();
    const input = Float32Array.from({ length: rows * 768 }, (_, index) =>
      Math.fround(Math.sin(index / 37) * 0.5 + Math.cos(index / 53) * 0.25));
    const positions = new Int32Array(rows * 2);
    for (let index = 0; index < rows; index += 1) {
      positions[index * 2] = Math.floor(index / 60);
      positions[index * 2 + 1] = index % 60;
    }
    const hidden = device.createBuffer({
      label: "Vision tile benchmark hidden",
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      label: "Vision tile benchmark readback",
      size: input.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const samples: Record<"oneRow" | "twoRows", number[]> = {
      oneRow: [],
      twoRows: [],
    };
    const outputs = new Map<1 | 2, Uint32Array>();

    const runMode = async (tile: 1 | 2, collect: boolean): Promise<void> => {
      const resources = await layer.createGemmaVisionLayerResources(
        device,
        hidden,
        rows,
        positions,
        weights,
        { hiddenProjectionRowsPerWorkgroup: tile },
      );
      try {
        device.queue.writeBuffer(hidden, 0, input);
        const encoder = device.createCommandEncoder();
        layer.encodeGemmaVisionLayer(encoder, resources);
        if (collect) {
          encoder.copyBufferToBuffer(hidden, 0, readback, 0, input.byteLength);
        }
        const started = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const elapsed = performance.now() - started;
        if (!collect) samples[tile === 1 ? "oneRow" : "twoRows"].push(elapsed);
        if (collect) {
          await readback.mapAsync(GPUMapMode.READ);
          outputs.set(tile, new Uint32Array(readback.getMappedRange().slice(0)));
          readback.unmap();
        }
      } finally {
        layer.destroyGemmaVisionLayerResources(resources);
      }
    };

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      await runMode(1, true);
      await runMode(2, true);
      for (let sample = 0; sample < samplesPerMode; sample += 1) {
        await runMode(sample % 2 === 0 ? 1 : 2, false);
        await runMode(sample % 2 === 0 ? 2 : 1, false);
      }
      const baseline = outputs.get(1)!;
      const optimized = outputs.get(2)!;
      let bitMismatches = 0;
      for (let index = 0; index < baseline.length; index += 1) {
        if (baseline[index] !== optimized[index]) bitMismatches += 1;
      }
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        rows,
        samples,
        bitMismatches,
        gpuError: internalError?.message ?? validationError?.message ?? null,
        adapterInfo: device.adapterInfo,
      };
    } finally {
      hidden.destroy();
      readback.destroy();
    }
  }, { rows: ROWS, samplesPerMode: SAMPLES_PER_MODE });

  const oneRow = distribution(result.samples.oneRow);
  const twoRows = distribution(result.samples.twoRows);
  const artifact = {
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    browser: await page.evaluate(() => navigator.userAgent),
    adapterInfo: result.adapterInfo,
    geometry: { rows: ROWS, hiddenSize: 768, layerIndex: 0 },
    methodology: {
      clock: "host-performance-now-around-submit-and-onSubmittedWorkDone",
      samplesPerMode: SAMPLES_PER_MODE,
      order: "alternating-balanced",
      setupExcluded: true,
      readbackExcluded: true,
    },
    exactness: { outputBitMismatches: result.bitMismatches },
    modes: {
      baselineOneRow: { samplesMs: result.samples.oneRow, ...oneRow },
      optimizedTwoRows: { samplesMs: result.samples.twoRows, ...twoRows },
    },
    comparison: {
      medianSpeedup: oneRow.medianMs / twoRows.medianMs,
      p95Speedup: oneRow.p95Ms / twoRows.p95Ms,
      promoted: twoRows.medianMs < oneRow.medianMs && twoRows.p95Ms < oneRow.p95Ms,
    },
  };
  const artifactPath = resolve("benchmarks/vision-dense-tile.chrome.json");
  await mkdir(resolve("benchmarks"), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");

  expect(result.gpuError).toBeNull();
  expect(result.bitMismatches).toBe(0);
  expect(result.samples.oneRow).toHaveLength(SAMPLES_PER_MODE);
  expect(result.samples.twoRows).toHaveLength(SAMPLES_PER_MODE);
  expect(artifact.comparison.promoted).toBe(true);
});

test("soaks mixed vision budgets, cancellation, recovery, and cleanup", async ({ page }) => {
  test.skip(!LIVE, "Set GEMMA_VISION_OPTIMIZATION=1 to run the live vision soak");
  test.setTimeout(600_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const weightsPath = "/src/model/gemma-vision-weights.ts";
    const devicePath = "/src/webgpu/device.ts";
    const inputPath = "/src/runtime/gemma-vision-input.ts";
    const imagePath = "/src/webgpu/vision-image.ts";
    const [{ PinnedSafetensorsSource }, { GemmaVisionWeightCache },
      { getWebGpuDevice }, input, vision] = await Promise.all([
      import(sourcePath),
      import(weightsPath),
      import(devicePath),
      import(inputPath),
      import(imagePath),
    ]);
    const source = await PinnedSafetensorsSource.open();
    const device = await getWebGpuDevice();
    const weightCache = new GemmaVisionWeightCache();
    const imageUrls = [
      "/examples/dolphin_capt_image.png",
      "/examples/the-mathematics-club-of-gottingen-1902.jpg",
    ];
    const blobs = await Promise.all(imageUrls.map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Vision soak image failed: ${response.status}`);
      return response.blob();
    }));
    const requests: Array<{ imageIndex: number; budget: 70 | 140 | 280 }> = [
      { imageIndex: 0, budget: 70 },
      { imageIndex: 1, budget: 140 },
      { imageIndex: 0, budget: 280 },
      { imageIndex: 1, budget: 70 },
      { imageIndex: 0, budget: 140 },
      { imageIndex: 1, budget: 280 },
    ];
    const completed: Array<{
      imageIndex: number;
      budget: number;
      patchCount: number;
      softTokenCount: number;
      layerExecutionMilliseconds: number;
    }> = [];
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      for (const request of requests) {
        const prepared = await input.prepareGemmaVisionImage(
          blobs[request.imageIndex],
          undefined,
          request.budget,
        );
        const encoded = await vision.encodeGemmaVisionImage(
          device,
          source,
          prepared,
          undefined,
          undefined,
          weightCache,
        );
        try {
          completed.push({
            ...request,
            patchCount: encoded.patchCount,
            softTokenCount: encoded.softTokenCount,
            layerExecutionMilliseconds: encoded.timing.layerExecutionMilliseconds,
          });
        } finally {
          vision.destroyGemmaVisionImageResources(encoded);
        }
      }
      const retainedAfterCompleted = weightCache.estimateRetainedMemory();
      const cancelledInput = await input.prepareGemmaVisionImage(blobs[0], undefined, 280);
      const controller = new AbortController();
      let cancellation: { name: string; message: string; completedLayers: number } | null = null;
      let completedLayers = 0;
      try {
        const encoded = await vision.encodeGemmaVisionImage(
          device,
          source,
          cancelledInput,
          (progress: { completedLayers: number }) => {
            completedLayers = progress.completedLayers;
            if (completedLayers === 2) {
              controller.abort(new DOMException("vision soak cancelled", "AbortError"));
            }
          },
          controller.signal,
          weightCache,
        );
        vision.destroyGemmaVisionImageResources(encoded);
      } catch (error) {
        cancellation = {
          name: (error as Error).name,
          message: (error as Error).message,
          completedLayers,
        };
      }
      const recoveryInput = await input.prepareGemmaVisionImage(blobs[1], undefined, 70);
      const recovery = await vision.encodeGemmaVisionImage(
        device,
        source,
        recoveryInput,
        undefined,
        undefined,
        weightCache,
      );
      const recoveryResult = {
        patchCount: recovery.patchCount,
        softTokenCount: recovery.softTokenCount,
      };
      vision.destroyGemmaVisionImageResources(recovery);
      await device.queue.onSubmittedWorkDone();
      const retainedAfterRecovery = weightCache.estimateRetainedMemory();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      weightCache.clear();
      return {
        completed,
        cancellation,
        recovery: recoveryResult,
        retainedAfterCompleted,
        retainedAfterRecovery,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } catch (error) {
      await device.popErrorScope();
      await device.popErrorScope();
      weightCache.clear();
      throw error;
    }
  });

  expect(result.gpuError).toBeNull();
  expect(result.completed).toHaveLength(6);
  expect(result.completed.map(({ softTokenCount }) => softTokenCount)).toEqual([
    66, 130, 276, 63, 128, 266,
  ]);
  expect(result.completed.every(({ layerExecutionMilliseconds }) =>
    layerExecutionMilliseconds > 0)).toBe(true);
  expect(result.cancellation).toEqual({
    name: "AbortError",
    message: "vision soak cancelled",
    completedLayers: 2,
  });
  expect(result.recovery.softTokenCount).toBe(63);
  expect(result.retainedAfterRecovery).toEqual(result.retainedAfterCompleted);
  expect(result.retainedAfterRecovery.loadedEntryCount).toBe(18);
  const artifactPath = resolve("benchmarks/vision-lifecycle-soak.chrome.json");
  await mkdir(resolve("benchmarks"), { recursive: true });
  await writeFile(artifactPath, `${JSON.stringify({
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    browser: await page.evaluate(() => navigator.userAgent),
    requests: result.completed,
    cancellation: result.cancellation,
    recovery: result.recovery,
    retainedVisionWeights: result.retainedAfterRecovery,
    gpuError: result.gpuError,
  }, null, 2)}\n`, "utf8");
});

function distribution(values: readonly number[]): { medianMs: number; p95Ms: number } {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
  };
}

function percentile(sorted: readonly number[], quantile: number): number {
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
