import { expect, test } from "@playwright/test";
import {
  counterbalancedPreference,
  familyPairedEffect,
  holmAdjust,
  judgeAgreement,
  randomControlTest,
  STATISTICS_LIMITS,
  summarizeOutcomes,
  type CounterbalancedPreferencePair,
  type FamilyPairedEffectOptions,
  type JudgePair,
  type PairedMetric,
  type PairedOutcome,
} from "../src/research/workbench/statistics";

const fastOptions = { seed: 42, bootstrapReplicates: 400, signFlipReplicates: 1000 };
const metric = (id: string, family: string, delta: number): PairedMetric => ({
  id, family, baseline: 0, intervention: delta,
});
const families = (deltas: readonly number[]): PairedMetric[] =>
  deltas.map((delta, index) => metric(`case-${index}`, `family-${index}`, delta));

test("outcomes count every transition and retain unknown-only families", () => {
  const summary = summarizeOutcomes([
    { id: "fixed", family: "a", baseline: false, intervention: true },
    { id: "broken", family: "a", baseline: true, intervention: false },
    { id: "correct", family: "b", baseline: true, intervention: true },
    { id: "wrong", family: "b", baseline: false, intervention: false },
    { id: "missing-first", family: "c", baseline: null, intervention: true },
    { id: "missing-second", family: "c", baseline: false, intervention: null },
    { id: "missing-both", family: "d", baseline: null, intervention: null },
  ]);
  expect(summary).toEqual({
    caseCount: 7, observedPairs: 4, familyCount: 4,
    fixed: 1, broken: 1, unchangedCorrect: 1, unchangedWrong: 1, unknown: 3,
  });
  expect(summarizeOutcomes([])).toEqual({
    caseCount: 0, observedPairs: 0, familyCount: 0,
    fixed: 0, broken: 0, unchangedCorrect: 0, unchangedWrong: 0, unknown: 0,
  });
});

test("outcomes reject duplicate IDs, empty families and invalid runtime values", () => {
  const row: PairedOutcome = { id: "x", family: "a", baseline: true, intervention: false };
  expect(() => summarizeOutcomes([row, { ...row, family: "b" }])).toThrow("Duplicate id");
  for (const family of ["", " \t "]) {
    expect(() => summarizeOutcomes([{ ...row, family }])).toThrow("family");
  }
  expect(() => summarizeOutcomes([{ ...row, id: " " }])).toThrow("id");
  for (const invalid of [0, 1, "true", undefined, Number.NaN, {}]) {
    expect(() => summarizeOutcomes([{ ...row, baseline: invalid }] as unknown as PairedOutcome[])).toThrow();
    expect(() => summarizeOutcomes([{ ...row, intervention: invalid }] as unknown as PairedOutcome[])).toThrow();
  }
  expect(() => summarizeOutcomes([null] as unknown as PairedOutcome[])).toThrow("object");
  expect(() => summarizeOutcomes(null as unknown as PairedOutcome[])).toThrow("array");
});

test("family effects weight families equally, not unequal numbers of rows", () => {
  const rows = [
    metric("a", "small", 6),
    ...Array.from({ length: 20 }, (_, i) => metric(`b-${i}`, "large", -2)),
  ];
  const result = familyPairedEffect(rows, fastOptions);
  expect(result.caseCount).toBe(21);
  expect(result.familyCount).toBe(2);
  expect(result.meanDelta).toBeCloseTo(2, 14);
  expect(result.familyEffects).toEqual([
    { family: "large", caseCount: 20, meanDelta: -2 },
    { family: "small", caseCount: 1, meanDelta: 6 },
  ]);
  expect(result.confidenceInterval95).toEqual([-2, 6]);
  expect(result.pValue).toBe(1);
  expect(result.exact).toBe(true);
  expect(result.lowPower).toBe(true);
  expect(result.replicates).toEqual({ bootstrap: 400, signFlip: 4 });
  expect(familyPairedEffect([rows[0], rows[1]], fastOptions).pValue).toBe(result.pValue);
  expect(result.metadata.unit).toBe("family");
  expect(result.metadata.interpretation).toContain("symmetric about zero");
  expect(result.metadata.interpretation).toContain("not automatically a valid randomization test");
});

test("within-family paired differences are averaged before the family mean", () => {
  const result = familyPairedEffect([
    { id: "a", family: "one", baseline: 10, intervention: 12 },
    { id: "b", family: "one", baseline: -3, intervention: 3 },
    { id: "c", family: "two", baseline: 100, intervention: 98 },
  ], fastOptions);
  expect(result.familyEffects.map((family) => family.meanDelta)).toEqual([4, -2]);
  expect(result.meanDelta).toBe(1);
});

test("a planted common effect has a point bootstrap interval and a discrete exact p-value", () => {
  const result = familyPairedEffect(families(new Array<number>(10).fill(3)), fastOptions);
  expect(result.meanDelta).toBe(3);
  expect(result.confidenceInterval95).toEqual([3, 3]);
  expect(result.pValue).toBe(2 / 1024);
  expect(result.exact).toBe(true);
  expect(result.lowPower).toBe(false);
  expect(result.replicates.signFlip).toBe(1024);
});

test("exact sign flips are two-sided, include ties and retain one-family discreteness", () => {
  for (const delta of [2, -2, 0]) {
    const result = familyPairedEffect(families([delta]), fastOptions);
    expect(result.meanDelta).toBe(delta);
    expect(result.confidenceInterval95).toBeNull();
    expect(result.pValue).toBe(1);
    expect(result.replicates).toEqual({ bootstrap: 0, signFlip: 2 });
  }
  expect(familyPairedEffect(families([1, 1]), fastOptions).pValue).toBe(0.5);
  expect(familyPairedEffect(families([-1, -1, -1]), fastOptions).pValue).toBe(0.25);
  expect(familyPairedEffect(families([1, 2, 3]), fastOptions).pValue).toBe(0.25);
  expect(familyPairedEffect(families([1, 1, 0]), fastOptions).pValue).toBe(0.5);
  expect(familyPairedEffect(families([1, -1]), fastOptions).pValue).toBe(1);
  const zeros = familyPairedEffect(families([0, 0, 0]), fastOptions);
  expect(zeros.confidenceInterval95).toEqual([0, 0]);
  expect(zeros.pValue).toBe(1);
});

test("bootstrap is seeded, order invariant, nonmutating and has a nondegenerate interval", () => {
  const rows = families([-7, -3, 2, 4, 8, 11]);
  const snapshot = structuredClone(rows);
  const result = familyPairedEffect(rows, fastOptions);
  expect(familyPairedEffect(rows, fastOptions)).toEqual(result);
  expect(familyPairedEffect([...rows].reverse(), fastOptions)).toEqual(result);
  expect(rows).toEqual(snapshot);
  expect(result.seed).toBe(42);
  expect(result.meanDelta).toBeCloseTo(2.5);
  expect(result.confidenceInterval95?.[0]).toBeLessThan(0);
  expect(result.confidenceInterval95?.[1]).toBeGreaterThan(5);
  expect(familyPairedEffect(rows, { ...fastOptions, seed: 17 }).confidenceInterval95)
    .not.toEqual(result.confidenceInterval95);
});

test("16 families are exact; 17 use repeatable corrected Monte Carlo", () => {
  const exact = familyPairedEffect(families(new Array<number>(16).fill(1)), fastOptions);
  expect(exact.exact).toBe(true);
  expect(exact.replicates.signFlip).toBe(65536);
  expect(exact.pValue).toBe(2 / 65536);
  const rows = families(Array.from({ length: 17 }, (_, i) => i % 3 === 0 ? -1 : 2));
  const result = familyPairedEffect(rows, fastOptions);
  expect(result.exact).toBe(false);
  expect(result.replicates.signFlip).toBe(fastOptions.signFlipReplicates);
  expect(result.pValue).toBeGreaterThanOrEqual(1 / 1001);
  expect(result.pValue).toBeLessThan(1);
  expect(result.pValue * 1001).toBeCloseTo(Math.round(result.pValue * 1001), 10);
  expect(familyPairedEffect(rows, fastOptions)).toEqual(result);
  expect(familyPairedEffect(rows, { ...fastOptions, bootstrapReplicates: 200 }).pValue).toBe(result.pValue);
  expect(familyPairedEffect(families(new Array<number>(17).fill(0)), fastOptions).pValue).toBe(1);
});

test("family effect rejects missing, nonfinite, overflowing and duplicate data", () => {
  const row = metric("x", "f", 1);
  expect(() => familyPairedEffect([])).toThrow("nonempty");
  expect(() => familyPairedEffect([row, { ...row, family: "other" }])).toThrow("Duplicate id");
  expect(() => familyPairedEffect([{ ...row, family: " " }])).toThrow("family");
  expect(() => familyPairedEffect([{ ...row, id: "" }])).toThrow("id");
  for (const invalid of [NaN, Infinity, -Infinity, null, undefined, "1"]) {
    expect(() => familyPairedEffect([{ ...row, baseline: invalid }] as unknown as PairedMetric[])).toThrow("finite");
    expect(() => familyPairedEffect([{ ...row, intervention: invalid }] as unknown as PairedMetric[])).toThrow("finite");
  }
  expect(() => familyPairedEffect([{ ...row, baseline: -Number.MAX_VALUE, intervention: Number.MAX_VALUE }]))
    .toThrow("paired delta");
});

test("options and total resampling work are bounded and strictly validated", () => {
  const rows = families([1, 2]);
  for (const seed of [-1, 0.5, 2 ** 32, Infinity, NaN, null, "4"]) {
    expect(() => familyPairedEffect(rows, { seed } as unknown as FamilyPairedEffectOptions)).toThrow("seed");
  }
  for (const count of [0, 1, -1, 2.5, Infinity, NaN, STATISTICS_LIMITS.maxReplicates + 1, null, "10"]) {
    for (const key of ["bootstrapReplicates", "signFlipReplicates"]) {
      expect(() => familyPairedEffect(rows, { [key]: count } as FamilyPairedEffectOptions)).toThrow(key);
    }
  }
  expect(() => familyPairedEffect(rows, { typo: 10 } as FamilyPairedEffectOptions)).toThrow("Unknown option");
  expect(() => familyPairedEffect(rows, null as unknown as FamilyPairedEffectOptions)).toThrow("object");
  expect(() => familyPairedEffect(families(new Array<number>(51).fill(1)), {
    bootstrapReplicates: STATISTICS_LIMITS.maxReplicates,
    signFlipReplicates: STATISTICS_LIMITS.maxReplicates,
  })).toThrow("budget");
  expect(() => summarizeOutcomes(new Array<PairedOutcome>(STATISTICS_LIMITS.maxRows + 1))).toThrow("limited");
  expect(familyPairedEffect(rows, { ...fastOptions, seed: 0 }).seed).toBe(0);
  expect(familyPairedEffect(rows, { ...fastOptions, seed: 0xffffffff }).seed).toBe(0xffffffff);
  const defaults = familyPairedEffect(rows);
  expect(defaults.seed).toBe(0x5eed1234);
  expect(defaults.replicates.bootstrap).toBe(2000);
});

test("scaled calculations handle very large and very small finite effects", () => {
  const large = Number.MAX_VALUE;
  const result = familyPairedEffect(families([large, large, large]), fastOptions);
  expect(result.meanDelta).toBe(large);
  expect(result.confidenceInterval95).toEqual([large, large]);
  expect(result.pValue).toBe(0.25);
  expect(familyPairedEffect(families([1e-300, 1e-300, 1e-300]), fastOptions).pValue).toBe(0.25);
  const controls = randomControlTest(large, [large / 2, large]);
  expect(Number.isFinite(controls.mean)).toBe(true);
  expect(controls.sd / large).toBeCloseTo(Math.SQRT1_2 / 2, 14);
  expect(() => randomControlTest(0, [-large, large])).toThrow("standard deviation");
});

test("random controls report sample dispersion and tie-inclusive empirical ranks", () => {
  const result = randomControlTest(2, [1, 2, 2, 3]);
  expect(result.n).toBe(4);
  expect(result.mean).toBe(2);
  expect(result.sd).toBeCloseTo(Math.sqrt(2 / 3), 14);
  expect(result.rank).toBe(4);
  expect(result.pValue).toBe(4 / 5);
  expect(result.min).toBe(1);
  expect(result.max).toBe(3);
  expect(result.metadata.interpretation).toContain("Not parametric significance");
  expect(randomControlTest(4, [1, 2, 3]).pValue).toBe(1 / 4);
  expect(randomControlTest(0, [1, 2, 3]).pValue).toBe(1);
  expect(randomControlTest(2, [2, 2])).toMatchObject({ mean: 2, sd: 0, rank: 3, pValue: 1 });
  expect(randomControlTest(0, [0, 0])).toMatchObject({ mean: 0, sd: 0, pValue: 1 });
});

test("random controls require two finite controls and a finite observed value", () => {
  for (const controls of [[], [1], [1, Infinity], [NaN, 1]]) {
    expect(() => randomControlTest(1, controls)).toThrow();
  }
  for (const invalid of [NaN, Infinity, -Infinity]) {
    expect(() => randomControlTest(invalid, [1, 2])).toThrow("finite");
  }
  expect(() => randomControlTest(1, [1, null] as unknown as number[])).toThrow("finite");
});

test("Holm adjustment is monotone, handles ties and restores original order", () => {
  const input = [0.01, 0.04, 0.03, 0.002];
  const adjusted = holmAdjust(input);
  [0.03, 0.06, 0.06, 0.008].forEach((value, i) => expect(adjusted[i]).toBeCloseTo(value, 14));
  expect(input).toEqual([0.01, 0.04, 0.03, 0.002]);
  expect(holmAdjust([0.02, 0.02, 0.6, 1, 0])).toEqual([0.08, 0.08, 1, 1, 0]);
  expect(holmAdjust([])).toEqual([]);
  expect(holmAdjust([0.3])).toEqual([0.3]);
  for (const invalid of [NaN, Infinity, -0.01, 1.01, null, "0.5"]) {
    expect(() => holmAdjust([invalid] as unknown as number[])).toThrow();
  }
});

test("judge agreement reports complete pairs separately from all abstentions", () => {
  const result = judgeAgreement([
    { id: "a", first: 10, second: 10 },
    { id: "b", first: 6.5, second: 7.5 },
    { id: "c", first: 3, second: 5 },
    { id: "d", first: null, second: 10 },
    { id: "e", first: 0, second: null },
    { id: "f", first: null, second: null },
  ]);
  expect(result).toMatchObject({
    caseCount: 6, completeCount: 3, abstentionCount: 3,
    firstAbstentionCount: 2, secondAbstentionCount: 2, bothAbstentionCount: 1,
    exactFraction: 1 / 3, withinOneFraction: 2 / 3, meanAbsoluteDifference: 1,
  });
  expect(result.metadata.interpretation).toContain("not independent evidence");
  for (const pairs of [[], [{ id: "a", first: null, second: null }]]) {
    expect(judgeAgreement(pairs)).toMatchObject({
      completeCount: 0, exactFraction: null, withinOneFraction: null, meanAbsoluteDifference: null,
    });
  }
});

test("judge agreement validates IDs and both scores without silently skipping bad abstentions", () => {
  const pair: JudgePair = { id: "x", first: 1, second: 2 };
  expect(() => judgeAgreement([pair, pair])).toThrow("Duplicate id");
  expect(() => judgeAgreement([{ ...pair, id: "" }])).toThrow("id");
  for (const invalid of [-1, 10.01, Infinity, NaN, undefined, "5"]) {
    expect(() => judgeAgreement([{ ...pair, first: invalid }] as unknown as JudgePair[])).toThrow();
    expect(() => judgeAgreement([{ ...pair, first: null, second: invalid }] as unknown as JudgePair[])).toThrow();
  }
});

test("counterbalanced preference compares candidate identity, not display position", () => {
  const result = counterbalancedPreference([
    { id: "a", originalWinner: "A", reversedWinner: "B" },
    { id: "b", originalWinner: "B", reversedWinner: "A" },
    { id: "c", originalWinner: "tie", reversedWinner: "tie" },
    { id: "d", originalWinner: "A", reversedWinner: "A" },
    { id: "e", originalWinner: "tie", reversedWinner: "B" },
    { id: "f", originalWinner: null, reversedWinner: "A" },
    { id: "g", originalWinner: "B", reversedWinner: null },
  ]);
  expect(result).toMatchObject({
    caseCount: 7, completeCount: 5, abstentionCount: 2,
    consistentCount: 3, inconsistentCount: 2, consistencyFraction: 3 / 5,
  });
  expect(result.metadata.interpretation).toContain("not a preference ranking");
  expect(counterbalancedPreference([]).consistencyFraction).toBeNull();
  expect(counterbalancedPreference([{ id: "x", originalWinner: null, reversedWinner: null }]))
    .toMatchObject({ abstentionCount: 1, consistencyFraction: null });
});

test("counterbalanced preference validates winners even when the other judge abstains", () => {
  const pair: CounterbalancedPreferencePair = { id: "x", originalWinner: "A", reversedWinner: "B" };
  expect(() => counterbalancedPreference([pair, pair])).toThrow("Duplicate id");
  for (const invalid of ["a", "C", 0, undefined]) {
    expect(() => counterbalancedPreference([
      { ...pair, originalWinner: invalid },
    ] as unknown as CounterbalancedPreferencePair[])).toThrow("winner");
    expect(() => counterbalancedPreference([
      { ...pair, originalWinner: null, reversedWinner: invalid },
    ] as unknown as CounterbalancedPreferencePair[])).toThrow("winner");
  }
});
