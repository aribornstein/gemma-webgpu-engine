import type { GemmaVisionImageSource } from "./gemma-vision-input";

export const GEMMA_VIDEO_MAX_DURATION_SECONDS = 60;
export const GEMMA_VIDEO_MAX_FRAMES = 60;
export const GEMMA_VIDEO_SAMPLE_RATE = 1;

export type GemmaVideoSource = Blob;

export interface GemmaVideoFrame {
  image: GemmaVisionImageSource;
  timestampSeconds: number;
}

export interface GemmaVideoFrames {
  durationSeconds: number;
  frames: readonly GemmaVideoFrame[];
}

export function formatGemmaVideoTimestamp(timestampSeconds: number): string {
  if (!Number.isFinite(timestampSeconds) || timestampSeconds < 0) {
    throw new Error("Gemma video timestamp must be finite and non-negative");
  }
  const wholeSeconds = Math.floor(timestampSeconds);
  return `${String(Math.floor(wholeSeconds / 60)).padStart(2, "0")}:` +
    String(wholeSeconds % 60).padStart(2, "0");
}

export function planGemmaVideoTimestamps(
  durationSeconds: number,
  maximumFrames = GEMMA_VIDEO_MAX_FRAMES,
): number[] {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    throw new Error("Gemma video duration must be positive and finite");
  }
  if (durationSeconds > GEMMA_VIDEO_MAX_DURATION_SECONDS) {
    throw new Error(`Gemma video duration must not exceed ${GEMMA_VIDEO_MAX_DURATION_SECONDS} seconds`);
  }
  if (!Number.isInteger(maximumFrames) || maximumFrames < 1 ||
      maximumFrames > GEMMA_VIDEO_MAX_FRAMES) {
    throw new Error(`Gemma video frame limit must be from 1 through ${GEMMA_VIDEO_MAX_FRAMES}`);
  }
  const frameCount = Math.min(maximumFrames, Math.max(1, Math.ceil(
    durationSeconds * GEMMA_VIDEO_SAMPLE_RATE,
  )));
  return Array.from(
    { length: frameCount },
    (_, index) => Math.min(
      durationSeconds,
      (index + 0.5) * durationSeconds / frameCount,
    ),
  );
}

export async function prepareGemmaVideo(
  source: GemmaVideoSource,
  signal?: AbortSignal,
  maximumFrames = GEMMA_VIDEO_MAX_FRAMES,
): Promise<GemmaVideoFrames> {
  signal?.throwIfAborted();
  if (!(source instanceof Blob) || source.size === 0) {
    throw new Error("Gemma video source must be a non-empty Blob");
  }
  const url = URL.createObjectURL(source);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;
  try {
    await waitForVideoEvent(video, "loadedmetadata", signal);
    const durationSeconds = await resolveVideoDuration(video, signal);
    const timestamps = planGemmaVideoTimestamps(durationSeconds, maximumFrames);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    if (canvas.width < 1 || canvas.height < 1) {
      throw new Error("Gemma video has invalid frame dimensions");
    }
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Gemma video frame canvas is unavailable");
    const frames: GemmaVideoFrame[] = [];
    for (const timestampSeconds of timestamps) {
      signal?.throwIfAborted();
      video.currentTime = timestampSeconds;
      await waitForVideoEvent(video, "seeked", signal);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push({
        image: context.getImageData(0, 0, canvas.width, canvas.height),
        timestampSeconds,
      });
    }
    return { durationSeconds, frames };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

function waitForVideoEvent(
  video: HTMLVideoElement,
  eventName: "loadedmetadata" | "seeked",
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener(eventName, onReady);
      video.removeEventListener("error", onError);
      signal?.removeEventListener("abort", onAbort);
    };
    const onReady = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("The browser could not decode the selected video"));
    };
    const onAbort = () => {
      cleanup();
      reject(signal?.reason ?? new DOMException("Video preparation aborted", "AbortError"));
    };
    video.addEventListener(eventName, onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function resolveVideoDuration(
  video: HTMLVideoElement,
  signal?: AbortSignal,
): Promise<number> {
  if (Number.isFinite(video.duration)) return video.duration;
  video.currentTime = Number.MAX_SAFE_INTEGER;
  await waitForVideoEvent(video, "seeked", signal);
  if (!Number.isFinite(video.duration)) {
    throw new Error("The browser could not determine the selected video duration");
  }
  return video.duration;
}