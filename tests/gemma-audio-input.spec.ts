import { expect, test } from "@playwright/test";
import {
  extractGemmaAudioFeatures,
  gemmaAudioFrameCount,
  gemmaAudioSoftTokenCount,
} from "../src/runtime/gemma-audio-input";

test("derives Gemma audio frame and soft-token geometry", () => {
  expect(gemmaAudioFrameCount(16_000)).toBe(99);
  expect(gemmaAudioFrameCount(480_000)).toBe(2_999);
  expect(gemmaAudioSoftTokenCount(new Uint8Array(99).fill(1))).toBe(25);
  expect(gemmaAudioSoftTokenCount(new Uint8Array(2_999).fill(1))).toBe(750);
});

test("extracts deterministic pinned log-mel features", () => {
  const waveform = Float32Array.from(
    { length: 16_000 },
    (_, index) => Math.fround(
      0.4 * Math.sin(2 * Math.PI * 440 * index / 16_000) +
      0.1 * Math.cos(2 * Math.PI * 880 * index / 16_000),
    ),
  );
  const first = extractGemmaAudioFeatures(waveform);
  const second = extractGemmaAudioFeatures(waveform);
  expect(first).toMatchObject({
    frameCount: 99,
    validFrameCount: 99,
    softTokenCount: 25,
    sampleCount: 16_000,
    paddedSampleCount: 16_000,
  });
  expect(new Uint32Array(first.features.buffer)).toEqual(
    new Uint32Array(second.features.buffer),
  );
  expect(first.features.every(Number.isFinite)).toBe(true);
  expect(Math.max(...first.features)).toBeGreaterThan(1);
  expect(Math.min(...first.features)).toBeLessThan(-6);
  const golden = [
    { frame: 0, bin: 0, value: -6.907755374908447 },
    { frame: 0, bin: 17, value: 0.5217311382293701 },
    { frame: 1, bin: 7, value: -3.7963125705718994 },
    { frame: 10, bin: 23, value: 2.6925153732299805 },
    { frame: 50, bin: 40, value: 1.8904809951782227 },
    { frame: 98, bin: 127, value: -6.90132474899292 },
  ];
  for (const { frame, bin, value } of golden) {
    expect(first.features[frame * 128 + bin]).toBeCloseTo(value, 5);
  }
});

test("zeros padded feature rows and rejects invalid PCM", () => {
  const result = extractGemmaAudioFeatures(new Float32Array(16_129));
  expect(result.paddedSampleCount).toBe(16_256);
  expect(result.mask.at(-1)).toBe(0);
  expect(Array.from(result.features.slice(-128))).toEqual(new Array(128).fill(0));
  expect(() => extractGemmaAudioFeatures(new Float32Array([Number.NaN]))).toThrow(
    "Gemma audio PCM must be finite",
  );
  expect(() => extractGemmaAudioFeatures(new Float32Array([0]), 48_000)).toThrow(
    "Gemma audio requires 16000 Hz PCM",
  );
});