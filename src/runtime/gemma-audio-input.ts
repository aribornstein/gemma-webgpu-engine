import FFT from "fft.js";

export const GEMMA_AUDIO_SAMPLE_RATE = 16_000;
export const GEMMA_AUDIO_FEATURE_SIZE = 128;
export const GEMMA_AUDIO_FRAME_LENGTH = 320;
export const GEMMA_AUDIO_HOP_LENGTH = 160;
export const GEMMA_AUDIO_FFT_LENGTH = 512;
export const GEMMA_AUDIO_MAX_SAMPLES = 480_000;
export const GEMMA_AUDIO_MAX_SOFT_TOKENS = 750;
export const GEMMA_AUDIO_PAD_MULTIPLE = 128;
export const GEMMA_AUDIO_MEL_FLOOR = 0.001;

const FREQUENCY_BINS = GEMMA_AUDIO_FFT_LENGTH / 2 + 1;
const MIN_FREQUENCY = 0;
const MAX_FREQUENCY = 8_000;

export interface GemmaAudioFeatures {
  identity?: string;
  features: Float32Array;
  mask: Uint8Array;
  frameCount: number;
  validFrameCount: number;
  softTokenCount: number;
  sampleCount: number;
  paddedSampleCount: number;
}

export interface GemmaAudioWaveformSource {
  waveform: Float32Array;
  samplingRate?: number;
}

export type GemmaAudioSource = Blob | GemmaAudioWaveformSource;

export async function prepareGemmaAudio(
  source: GemmaAudioSource,
  signal?: AbortSignal,
): Promise<GemmaAudioFeatures> {
  signal?.throwIfAborted();
  let waveform: Float32Array;
  let samplingRate: number;
  if (source instanceof Blob) {
    const bytes = await source.arrayBuffer();
    signal?.throwIfAborted();
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(bytes.slice(0));
      samplingRate = decoded.sampleRate;
      waveform = mixGemmaAudioChannels(decoded);
    } finally {
      await context.close();
    }
  } else {
    waveform = new Float32Array(source.waveform);
    samplingRate = source.samplingRate ?? GEMMA_AUDIO_SAMPLE_RATE;
  }
  signal?.throwIfAborted();
  const resampled = samplingRate === GEMMA_AUDIO_SAMPLE_RATE
    ? waveform
    : await resampleGemmaAudio(waveform, samplingRate, signal);
  const used = resampled.subarray(0, GEMMA_AUDIO_MAX_SAMPLES);
  const identity = await gemmaAudioContentIdentity(used);
  signal?.throwIfAborted();
  return { ...extractGemmaAudioFeatures(used), identity };
}

export async function gemmaAudioContentIdentity(waveform: Float32Array): Promise<string> {
  const header = new Uint32Array([GEMMA_AUDIO_SAMPLE_RATE, waveform.length]);
  const payload = new Uint8Array(header.byteLength + waveform.byteLength);
  payload.set(new Uint8Array(header.buffer));
  payload.set(new Uint8Array(waveform.buffer, waveform.byteOffset, waveform.byteLength),
    header.byteLength);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", payload));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function extractGemmaAudioFeatures(
  waveform: Float32Array,
  samplingRate = GEMMA_AUDIO_SAMPLE_RATE,
): GemmaAudioFeatures {
  if (samplingRate !== GEMMA_AUDIO_SAMPLE_RATE) {
    throw new Error(`Gemma audio requires ${GEMMA_AUDIO_SAMPLE_RATE} Hz PCM`);
  }
  if (waveform.length < 1) throw new Error("Gemma audio requires at least one sample");
  for (const sample of waveform) {
    if (!Number.isFinite(sample)) throw new Error("Gemma audio PCM must be finite");
  }

  const sampleCount = Math.min(waveform.length, GEMMA_AUDIO_MAX_SAMPLES);
  const paddedSampleCount = roundUp(sampleCount, GEMMA_AUDIO_PAD_MULTIPLE);
  const semicausalPadding = GEMMA_AUDIO_FRAME_LENGTH / 2;
  const framedSampleCount = paddedSampleCount + semicausalPadding;
  const frameSizeForUnfold = GEMMA_AUDIO_FRAME_LENGTH + 1;
  const frameCount = Math.max(
    0,
    Math.floor((framedSampleCount - frameSizeForUnfold) / GEMMA_AUDIO_HOP_LENGTH) + 1,
  );
  const features = new Float32Array(frameCount * GEMMA_AUDIO_FEATURE_SIZE);
  const mask = new Uint8Array(frameCount);
  const window = createPeriodicHannWindow();
  const melFilters = createHtkMelFilters();
  const fft = new FFT(GEMMA_AUDIO_FFT_LENGTH);
  const fftInput = new Float64Array(GEMMA_AUDIO_FFT_LENGTH);
  const fftOutput = fft.createComplexArray() as number[];

  for (let frame = 0; frame < frameCount; frame += 1) {
    fftInput.fill(0);
    const frameStart = frame * GEMMA_AUDIO_HOP_LENGTH - semicausalPadding;
    for (let sample = 0; sample < GEMMA_AUDIO_FRAME_LENGTH; sample += 1) {
      const waveformIndex = frameStart + sample;
      const value = waveformIndex >= 0 && waveformIndex < sampleCount
        ? waveform[waveformIndex]
        : 0;
      fftInput[sample] = value * window[sample];
    }
    fft.realTransform(fftOutput, fftInput);
    const frameEnd = frame * GEMMA_AUDIO_HOP_LENGTH + GEMMA_AUDIO_FRAME_LENGTH;
    const valid = frameEnd >= semicausalPadding &&
      frameEnd - semicausalPadding < sampleCount;
    mask[frame] = valid ? 1 : 0;
    if (!valid) continue;
    for (let mel = 0; mel < GEMMA_AUDIO_FEATURE_SIZE; mel += 1) {
      let projected = 0;
      const filterOffset = mel * FREQUENCY_BINS;
      for (let bin = 0; bin < FREQUENCY_BINS; bin += 1) {
        const real = fftOutput[bin * 2];
        const imaginary = fftOutput[bin * 2 + 1];
        projected += Math.hypot(real, imaginary) * melFilters[filterOffset + bin];
      }
      features[frame * GEMMA_AUDIO_FEATURE_SIZE + mel] = Math.fround(
        Math.log(projected + GEMMA_AUDIO_MEL_FLOOR),
      );
    }
  }

  const validFrameCount = mask.reduce((sum, value) => sum + value, 0);
  return {
    features,
    mask,
    frameCount,
    validFrameCount,
    softTokenCount: gemmaAudioSoftTokenCount(mask),
    sampleCount,
    paddedSampleCount,
  };
}

export function gemmaAudioSoftTokenCount(frameMask: Uint8Array): number {
  let mask = frameMask;
  for (let layer = 0; layer < 2; layer += 1) {
    const outputLength = Math.floor((mask.length + 1) / 2);
    const output = new Uint8Array(outputLength);
    for (let index = 0; index < outputLength; index += 1) {
      output[index] = mask[index * 2];
    }
    mask = output;
  }
  return Math.min(
    mask.reduce((sum, value) => sum + value, 0),
    GEMMA_AUDIO_MAX_SOFT_TOKENS,
  );
}

export function gemmaAudioFrameCount(sampleCount: number): number {
  if (!Number.isInteger(sampleCount) || sampleCount < 1) {
    throw new Error("Gemma audio sample count must be a positive integer");
  }
  const capped = Math.min(sampleCount, GEMMA_AUDIO_MAX_SAMPLES);
  const padded = roundUp(capped, GEMMA_AUDIO_PAD_MULTIPLE);
  return Math.max(
    0,
    Math.floor(
      (padded + GEMMA_AUDIO_FRAME_LENGTH / 2 - GEMMA_AUDIO_FRAME_LENGTH - 1) /
      GEMMA_AUDIO_HOP_LENGTH,
    ) + 1,
  );
}

function createPeriodicHannWindow(): Float64Array {
  return Float64Array.from(
    { length: GEMMA_AUDIO_FRAME_LENGTH },
    (_, index) => 0.5 - 0.5 * Math.cos(2 * Math.PI * index / GEMMA_AUDIO_FRAME_LENGTH),
  );
}

function createHtkMelFilters(): Float64Array {
  const melMinimum = hertzToMel(MIN_FREQUENCY);
  const melMaximum = hertzToMel(MAX_FREQUENCY);
  const centers = Float64Array.from(
    { length: GEMMA_AUDIO_FEATURE_SIZE + 2 },
    (_, index) => melToHertz(
      melMinimum + (melMaximum - melMinimum) * index / (GEMMA_AUDIO_FEATURE_SIZE + 1),
    ),
  );
  const filters = new Float64Array(GEMMA_AUDIO_FEATURE_SIZE * FREQUENCY_BINS);
  for (let mel = 0; mel < GEMMA_AUDIO_FEATURE_SIZE; mel += 1) {
    const left = centers[mel];
    const center = centers[mel + 1];
    const right = centers[mel + 2];
    for (let bin = 0; bin < FREQUENCY_BINS; bin += 1) {
      const frequency = bin * GEMMA_AUDIO_SAMPLE_RATE / GEMMA_AUDIO_FFT_LENGTH;
      const down = (frequency - left) / (center - left);
      const up = (right - frequency) / (right - center);
      filters[mel * FREQUENCY_BINS + bin] = Math.max(0, Math.min(down, up));
    }
  }
  return filters;
}

function hertzToMel(frequency: number): number {
  return 2595 * Math.log10(1 + frequency / 700);
}

function melToHertz(mels: number): number {
  return 700 * (10 ** (mels / 2595) - 1);
}

function roundUp(value: number, multiple: number): number {
  return Math.ceil(value / multiple) * multiple;
}

function mixGemmaAudioChannels(audio: AudioBuffer): Float32Array {
  const waveform = new Float32Array(audio.length);
  for (let channel = 0; channel < audio.numberOfChannels; channel += 1) {
    const values = audio.getChannelData(channel);
    for (let index = 0; index < values.length; index += 1) {
      waveform[index] += values[index] / audio.numberOfChannels;
    }
  }
  return waveform;
}

async function resampleGemmaAudio(
  waveform: Float32Array,
  samplingRate: number,
  signal?: AbortSignal,
): Promise<Float32Array> {
  if (!Number.isFinite(samplingRate) || samplingRate <= 0) {
    throw new Error("Gemma audio sampling rate must be positive");
  }
  const outputLength = Math.max(1, Math.round(
    waveform.length * GEMMA_AUDIO_SAMPLE_RATE / samplingRate,
  ));
  const context = new OfflineAudioContext(1, outputLength, GEMMA_AUDIO_SAMPLE_RATE);
  const buffer = context.createBuffer(1, waveform.length, samplingRate);
  buffer.getChannelData(0).set(waveform);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  signal?.throwIfAborted();
  return new Float32Array(rendered.getChannelData(0));
}