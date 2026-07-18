import { expect, test } from "@playwright/test";
import {
  formatGemmaVideoTimestamp,
  GEMMA_VIDEO_MAX_DURATION_SECONDS,
  GEMMA_VIDEO_MAX_FRAMES,
  planGemmaVideoTimestamps,
} from "../src/runtime/gemma-video-input";

test("plans deterministic one-frame-per-second video samples", () => {
  expect(planGemmaVideoTimestamps(4)).toEqual([0.5, 1.5, 2.5, 3.5]);
  expect(planGemmaVideoTimestamps(0.25)).toEqual([0.125]);
});

test("uniformly caps long videos at the supported frame limit", () => {
  const timestamps = planGemmaVideoTimestamps(GEMMA_VIDEO_MAX_DURATION_SECONDS);
  expect(timestamps).toHaveLength(GEMMA_VIDEO_MAX_FRAMES);
  expect(timestamps[0]).toBe(0.5);
  expect(timestamps.at(-1)).toBe(59.5);
});

test("rejects invalid video sampling geometry", () => {
  expect(() => planGemmaVideoTimestamps(0)).toThrow("duration");
  expect(() => planGemmaVideoTimestamps(Number.POSITIVE_INFINITY)).toThrow("duration");
  expect(() => planGemmaVideoTimestamps(GEMMA_VIDEO_MAX_DURATION_SECONDS + 0.01))
    .toThrow("60 seconds");
  expect(() => planGemmaVideoTimestamps(2, 0)).toThrow("frame limit");
  expect(() => planGemmaVideoTimestamps(2, GEMMA_VIDEO_MAX_FRAMES + 1)).toThrow("frame limit");
});

test("formats processor-compatible video timestamps", () => {
  expect(formatGemmaVideoTimestamp(0.5)).toBe("00:00");
  expect(formatGemmaVideoTimestamp(59.9)).toBe("00:59");
  expect(formatGemmaVideoTimestamp(60)).toBe("01:00");
  expect(() => formatGemmaVideoTimestamp(-1)).toThrow("timestamp");
});

test("decodes frames from a browser MediaRecorder clip", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const cameraModulePath = "/src/runtime/gemma-camera.ts";
    const videoModulePath = "/src/runtime/gemma-video-input.ts";
    const [{ startGemmaCameraCapture }, { prepareGemmaVideo }] = await Promise.all([
      import(cameraModulePath),
      import(videoModulePath),
    ]);
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 48;
    const context = canvas.getContext("2d")!;
    let frame = 0;
    const timer = window.setInterval(() => {
      context.fillStyle = frame++ % 2 === 0 ? "#e65332" : "#16857a";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }, 50);
    const capture = startGemmaCameraCapture(canvas.captureStream(10));
    await new Promise((resolve) => setTimeout(resolve, 1_100));
    const blob = await capture.stop();
    window.clearInterval(timer);
    const previewUrl = URL.createObjectURL(blob);
    const preview = document.createElement("video");
    preview.muted = true;
    preview.src = previewUrl;
    await new Promise<void>((resolve, reject) => {
      preview.addEventListener("loadedmetadata", () => resolve(), { once: true });
      preview.addEventListener("error", () => reject(preview.error), { once: true });
    });
    await preview.play();
    await new Promise((resolve) => setTimeout(resolve, 100));
    const previewDuration = preview.duration;
    const previewTime = preview.currentTime;
    preview.pause();
    preview.removeAttribute("src");
    preview.load();
    URL.revokeObjectURL(previewUrl);
    const prepared = await prepareGemmaVideo(blob);
    return {
      blobSize: blob.size,
      previewDuration,
      previewTime,
      durationSeconds: prepared.durationSeconds,
      frameCount: prepared.frames.length,
      width: prepared.frames[0]?.image.width,
      height: prepared.frames[0]?.image.height,
    };
  });

  expect(result.blobSize).toBeGreaterThan(0);
  expect(Number.isFinite(result.previewDuration)).toBe(true);
  expect(result.previewDuration).toBeGreaterThan(0);
  expect(result.previewTime).toBeGreaterThan(0);
  expect(result.durationSeconds).toBeGreaterThan(0);
  expect(result.frameCount).toBeGreaterThan(0);
  expect(result.width).toBe(64);
  expect(result.height).toBe(48);
});