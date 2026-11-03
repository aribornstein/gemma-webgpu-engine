import { expect, test } from "@playwright/test";
import { DEFAULT_DECODING_CONFIG, type DecodingConfig } from "../src/runtime/decoding";
import { sampleToken, SeededRandom } from "../src/runtime/sampling";
import { referenceSampleToken } from "./support/reference-sampling";

const VOCAB = 262_144;

function makeLogits(kind: string, seed: number): Float32Array {
  const random = new SeededRandom(seed);
  const logits = new Float32Array(VOCAB);
  for (let index = 0; index < VOCAB; index += 1) {
    const u = random.next() || 1e-9;
    logits[index] = 3 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random.next());
  }
  if (kind === "ties") for (let index = 0; index < VOCAB; index += 1) logits[index] = Math.round(logits[index] * 4) / 4;
  if (kind === "masked") for (let index = 0; index < VOCAB; index += 1) if (index % 97) logits[index] = -Infinity;
  if (kind === "peaked") logits[12_345] = 40;
  if (kind === "signed-zero") for (let index = 0; index < VOCAB; index += 3) logits[index] = index % 2 ? -0 : 0;
  if (kind === "nan") logits[77] = Number.NaN;
  return logits;
}

function outcome(select: () => number): number | string {
  try {
    return select();
  } catch (error) {
    return `throws ${error instanceof Error ? error.name : String(error)}`;
  }
}

const configs: DecodingConfig[] = [
  { ...DEFAULT_DECODING_CONFIG, temperature: 0.8, topK: 40, topP: 0.95 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 1, topK: 64, topP: 0.95 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 0.7, topK: 0, topP: 0.9 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 1.3, topK: 0, topP: 1, minP: 0.05 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 0.9, topK: 100, topP: 1, typicalP: 0.8 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 0.5, topK: 5, topP: 0.5, repetitionPenalty: 1.3, frequencyPenalty: 0.4, presencePenalty: 0.2 },
  { ...DEFAULT_DECODING_CONFIG, temperature: 0, repetitionPenalty: 1.2 },
];

for (const kind of ["normal", "ties", "masked", "peaked", "signed-zero", "nan"]) {
  test(`radix sampler matches the full-sort reference for ${kind} logits`, () => {
    const logits = makeLogits(kind, kind.length * 31 + 7);
    const history = Array.from({ length: 200 }, (_, index) => (index * 7919 + 13) % VOCAB);
    for (const config of configs) {
      for (const seed of [1, 2]) {
        const expectedRandom = new SeededRandom(seed);
        const actualRandom = new SeededRandom(seed);
        const expected = outcome(() => referenceSampleToken(logits, history, config, () => expectedRandom.next()));
        const actual = outcome(() => sampleToken(logits, history, config, () => actualRandom.next()));
        expect(actual, JSON.stringify({ config, seed })).toBe(expected);
        expect(actualRandom.next()).toBe(expectedRandom.next());
      }
    }
  });
}
