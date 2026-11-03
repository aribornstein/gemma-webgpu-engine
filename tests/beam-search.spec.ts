import { expect, test } from "@playwright/test";
import {
  bannedNgramTokens,
  beamHypothesisScore,
  resolveBeamSearchOptions,
  runBeamSearch,
  topLogProbabilityIndices,
  validateBeamSearchOptions,
  type BeamSearchOptions,
} from "../src/runtime/beam-search";

const END = 0;

/** Toy language model: next-token log-probabilities depend on the generated prefix. */
function toyModel(table: Record<string, number[]>, vocab: number) {
  return (prefix: readonly number[]): number[] => {
    const row = table[prefix.join(",")];
    if (row) return row.map(Math.log);
    return Array.from({ length: vocab }, (_, token) => Math.log(token === END ? 0.97 : 0.03 / (vocab - 1)));
  };
}

async function search(
  model: (prefix: readonly number[]) => number[],
  options: Partial<BeamSearchOptions>,
  maxNewTokens: number,
) {
  let beams: number[][] = [[]];
  const advances: { parents: number[]; tokenIds: number[] }[] = [];
  const result = await runBeamSearch({
    options: resolveBeamSearchOptions(options),
    maxNewTokens,
    initialLogits: model([]),
    isTerminal: (tokenId) => tokenId === END,
    logProbabilities: (logits) => logits,
    advance: async (parents, tokenIds) => {
      advances.push({ parents: [...parents], tokenIds: [...tokenIds] });
      beams = parents.map((parent, index) => [...beams[parent], tokenIds[index]]);
      return beams.map((prefix) => model(prefix));
    },
  });
  return { result, advances };
}

test("one beam matches greedy decoding", async () => {
  const model = toyModel({
    "": [0, 0.6, 0.4],
    "1": [0.1, 0.1, 0.8],
    "1,2": [0.9, 0.05, 0.05],
  }, 3);
  const { result } = await search(model, { numBeams: 1 }, 8);
  expect(result.best.tokenIds).toEqual([1, 2]);
  expect(result.best.terminalTokenId).toBe(END);
});

test("beam search recovers a sequence that greedy decoding misses", async () => {
  // Greedy takes 1 (0.6) and then faces a flat distribution; 2 (0.4) leads to a confident path.
  const model = toyModel({
    "": [0, 0.6, 0.4],
    "1": [0.34, 0.33, 0.33],
    "2": [0.02, 0.01, 0.97],
    "2,2": [0.98, 0.01, 0.01],
    "1,1": [0.98, 0.01, 0.01],
    "1,2": [0.98, 0.01, 0.01],
  }, 3);
  const greedy = await search(model, { numBeams: 1 }, 6);
  const beam = await search(model, { numBeams: 2, earlyStopping: true }, 6);
  expect(greedy.result.best.tokenIds).toEqual([1]);
  expect(beam.result.best.tokenIds).toEqual([2, 2]);
  expect(beam.result.best.sumLogProbability).toBeCloseTo(Math.log(0.4 * 0.97 * 0.98), 12);
  expect(beam.result.hypotheses.length).toBe(2);
  expect(beam.result.hypotheses[0].score).toBeGreaterThanOrEqual(beam.result.hypotheses[1].score);
});

test("reorders beams from their parents and stops at the length limit", async () => {
  const uniform = () => [Math.log(0.01), Math.log(0.5), Math.log(0.49)];
  const { result, advances } = await search(uniform, { numBeams: 2 }, 3);
  expect(advances).toEqual([
    { parents: [0, 0], tokenIds: [1, 2] },
    { parents: [0, 0], tokenIds: [1, 2] },
  ]);
  expect(result.best.tokenIds).toEqual([1, 1, 1]);
  expect(result.best.terminalTokenId).toBeNull();
  expect(result.steps).toBe(3);
});

test("length penalty changes which finished hypothesis ranks first", async () => {
  const model = toyModel({
    "": [0.5, 0.5, 0],
    "1": [0.0, 0.0, 1],
    "1,2": [0.9, 0, 0.1],
  }, 3);
  const short = await search(model, { numBeams: 2, lengthPenalty: 0, earlyStopping: true }, 4);
  expect(short.result.best.tokenIds).toEqual([]);
  expect(beamHypothesisScore(Math.log(0.5 * 0.9), 3, 2)).toBeGreaterThan(beamHypothesisScore(Math.log(0.5), 1, 2));
  const long = await search(model, { numBeams: 2, lengthPenalty: 2, earlyStopping: false }, 4);
  expect(long.result.best.tokenIds).toEqual([1, 2]);
});

test("selects top log-probabilities with deterministic ties and skips masked tokens", () => {
  expect(topLogProbabilityIndices([1, 3, 3, Number.NEGATIVE_INFINITY, 2], 3)).toEqual([1, 2, 4]);
  expect(topLogProbabilityIndices([Number.NEGATIVE_INFINITY, 0], 4)).toEqual([1]);
});

test("validates beam search options", () => {
  expect(validateBeamSearchOptions({ numBeams: 0, lengthPenalty: Number.NaN })).toEqual([
    "numBeams must be an integer from 1 to 16",
    "lengthPenalty must be finite",
  ]);
  expect(() => resolveBeamSearchOptions({ numBeams: 17 })).toThrow("numBeams");
  expect(validateBeamSearchOptions({ noRepeatNgramSize: 1.5 })).toEqual([
    "noRepeatNgramSize must be a non-negative integer",
  ]);
  expect(resolveBeamSearchOptions({})).toEqual({
    numBeams: 4,
    lengthPenalty: 1,
    earlyStopping: false,
    noRepeatNgramSize: 0,
  });
});

test("bans tokens that would repeat an n-gram", () => {
  expect([...bannedNgramTokens([5, 6, 7, 5, 6], 3)]).toEqual([7]);
  expect([...bannedNgramTokens([5, 6, 5, 8, 5], 2)].sort()).toEqual([6, 8]);
  expect([...bannedNgramTokens([5, 6], 1)].sort()).toEqual([5, 6]);
  expect(bannedNgramTokens([5, 6, 7], 0).size).toBe(0);
  expect(bannedNgramTokens([5], 3).size).toBe(0);
});

test("n-gram blocking breaks a repetition loop", async () => {
  // Token 1 is always most likely, so plain search repeats it until the length limit.
  const loop = () => [Math.log(0.1), Math.log(0.6), Math.log(0.3)];
  const plain = await search(loop, { numBeams: 2 }, 4);
  expect(plain.result.best.tokenIds).toEqual([1, 1, 1, 1]);
  const blocked = await search(loop, { numBeams: 2, noRepeatNgramSize: 2 }, 4);
  const pairs = new Set<string>();
  const tokens = blocked.result.best.tokenIds;
  for (let index = 0; index + 1 < tokens.length; index += 1) {
    const pair = `${tokens[index]},${tokens[index + 1]}`;
    expect(pairs.has(pair)).toBe(false);
    pairs.add(pair);
  }
});
