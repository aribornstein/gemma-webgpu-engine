import { expect, test } from "@playwright/test";
import {
  GEMMA_VISION_MAX_PATCHES,
  GEMMA_VISION_PATCH_DIMENSION,
  gemmaVisionTargetSize,
  patchifyGemmaVisionRgb,
} from "../src/runtime/gemma-vision-input";

test("matches Gemma 4 aspect-ratio preserving image geometry", () => {
  expect(gemmaVisionTargetSize(768, 768)).toEqual([768, 768]);
  expect(gemmaVisionTargetSize(600, 1200)).toEqual([528, 1104]);
  expect(gemmaVisionTargetSize(1200, 600)).toEqual([1104, 528]);
  expect(gemmaVisionTargetSize(1, 1000)).toEqual([48, 13_440]);
});

test("patchifies HWC RGB pixels and pads positions exactly", () => {
  const pixels = Float32Array.from(
    { length: 16 * 32 * 3 },
    (_, index) => index,
  );
  const result = patchifyGemmaVisionRgb(pixels, 16, 32);
  expect(result).toMatchObject({
    patchRows: 1,
    patchColumns: 2,
    patchCount: 2,
    softTokenCount: 0,
  });
  expect(result.patches.length).toBe(GEMMA_VISION_MAX_PATCHES * GEMMA_VISION_PATCH_DIMENSION);
  expect(Array.from(result.positions.slice(0, 6))).toEqual([0, 0, 1, 0, -1, -1]);
  expect(Array.from(result.patches.slice(0, 6))).toEqual([0, 1, 2, 3, 4, 5]);
  expect(Array.from(result.patches.slice(GEMMA_VISION_PATCH_DIMENSION,
    GEMMA_VISION_PATCH_DIMENSION + 6))).toEqual([48, 49, 50, 51, 52, 53]);
});

test("decodes and resizes a browser image into normalized vision patches", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-vision-input.ts";
    const { prepareGemmaVisionImage } = await import(modulePath);
    const bytes = new Uint8ClampedArray(48 * 48 * 4);
    for (let index = 0; index < bytes.length; index += 4) {
      bytes[index] = 255;
      bytes[index + 1] = 128;
      bytes[index + 2] = 0;
      bytes[index + 3] = 255;
    }
    const prepared = await prepareGemmaVisionImage(new ImageData(bytes, 48, 48));
    return {
      patchRows: prepared.patchRows,
      patchColumns: prepared.patchColumns,
      patchCount: prepared.patchCount,
      softTokenCount: prepared.softTokenCount,
      firstPixel: Array.from(prepared.patches.slice(0, 3)),
      firstPositions: Array.from(prepared.positions.slice(0, 6)),
    };
  });

  expect(result).toEqual({
    patchRows: 48,
    patchColumns: 48,
    patchCount: 2304,
    softTokenCount: 256,
    firstPixel: [1, Math.fround(128 / 255), 0],
    firstPositions: [0, 0, 1, 0, 2, 0],
  });
});

test("prepares the maximum 2,520-patch image geometry", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-vision-input.ts";
    const { prepareGemmaVisionImage } = await import(modulePath);
    const prepared = await prepareGemmaVisionImage(new ImageData(1, 1000));
    return {
      patchRows: prepared.patchRows,
      patchColumns: prepared.patchColumns,
      patchCount: prepared.patchCount,
      softTokenCount: prepared.softTokenCount,
    };
  });

  expect(result).toEqual({
    patchRows: 840,
    patchColumns: 3,
    patchCount: 2520,
    softTokenCount: 280,
  });
});

test("rejects cancelled image preparation before browser decoding", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/runtime/gemma-vision-input.ts";
    const { prepareGemmaVisionImage } = await import(modulePath);
    const controller = new AbortController();
    controller.abort(new DOMException("fixture cancelled", "AbortError"));
    try {
      await prepareGemmaVisionImage(new ImageData(48, 48), controller.signal);
      return null;
    } catch (error) {
      return { name: (error as Error).name, message: (error as Error).message };
    }
  });

  expect(result).toEqual({ name: "AbortError", message: "fixture cancelled" });
});