/** Bounds apply before resampling; operations count family draws/sign assignments. */
export const STATISTICS_LIMITS = Object.freeze({
  maxRows: 100_000,
  maxReplicates: 100_000,
  maxResamplingOperations: 10_000_000,
  exactFamilyLimit: 16,
});

export interface PairedOutcome {
  id: string;
  family: string;
  baseline: boolean | null;
  intervention: boolean | null;
}

export interface OutcomeSummary {
  caseCount: number;
  observedPairs: number;
  familyCount: number;
  fixed: number;
  broken: number;
  unchangedCorrect: number;
  unchangedWrong: number;
  unknown: number;
}

export interface PairedMetric {
  id: string;
  family: string;
  baseline: number;
  intervention: number;
}

export interface FamilyPairedEffectOptions {
  /** Unsigned 32-bit seed; defaults to 0x5eed1234. */
  seed?: number;
  /** Integer in [2, maxReplicates]; defaults to 2,000. */
  bootstrapReplicates?: number;
  /** Integer in [2, maxReplicates]; defaults to 10,000; unused for exact tests. */
  signFlipReplicates?: number;
}

export interface FamilyEffect {
  family: string;
  caseCount: number;
  meanDelta: number;
}

export interface FamilyPairedEffectResult {
  caseCount: number;
  familyCount: number;
  familyEffects: FamilyEffect[];
  meanDelta: number;
  confidenceInterval95: [number, number] | null;
  pValue: number;
  exact: boolean;
  lowPower: boolean;
  seed: number;
  /** Actual evaluations: zero bootstrap draws for one family, 2^families for exact tests. */
  replicates: { bootstrap: number; signFlip: number };
  metadata: {
    unit: "family";
    alternative: "two-sided";
    confidenceLevel: 0.95;
    bootstrapMethod: "percentile";
    interpretation: string;
  };
}

export interface RandomControlResult {
  n: number;
  mean: number;
  /** Sample standard deviation (n - 1 denominator). */
  sd: number;
  /** Conservative upper-tail rank: 1 + number of controls >= observed. */
  rank: number;
  pValue: number;
  min: number;
  max: number;
  metadata: { interpretation: string };
}

export interface JudgePair {
  id: string;
  first: number | null;
  second: number | null;
}

export interface JudgeAgreementResult {
  caseCount: number;
  completeCount: number;
  /** Pairs with at least one abstention; not the sum of individual abstentions. */
  abstentionCount: number;
  firstAbstentionCount: number;
  secondAbstentionCount: number;
  bothAbstentionCount: number;
  exactFraction: number | null;
  withinOneFraction: number | null;
  meanAbsoluteDifference: number | null;
  metadata: { interpretation: string };
}

export type PreferenceWinner = "A" | "B" | "tie" | null;

export interface CounterbalancedPreferencePair {
  id: string;
  originalWinner: PreferenceWinner;
  /** Position label in the reversed presentation, not the original candidate identity. */
  reversedWinner: PreferenceWinner;
}

export interface CounterbalancedPreferenceResult {
  caseCount: number;
  completeCount: number;
  abstentionCount: number;
  consistentCount: number;
  inconsistentCount: number;
  consistencyFraction: number | null;
  metadata: { interpretation: string };
}

function validateArray(values: unknown, label: string): void {
  if (!Array.isArray(values)) throw new TypeError(`${label} must be an array`);
  if (values.length > STATISTICS_LIMITS.maxRows) {
    throw new RangeError(`${label} is limited to ${STATISTICS_LIMITS.maxRows} entries`);
  }
}

function validateRecord(value: unknown, label: string): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
}

function validateId(id: unknown, seen: Set<string>): void {
  if (typeof id !== "string" || id.trim().length === 0) throw new TypeError("id must be a nonempty string");
  if (seen.has(id)) throw new Error(`Duplicate id: ${id}`);
  seen.add(id);
}

function validateFamily(family: unknown): void {
  if (typeof family !== "string" || family.trim().length === 0) {
    throw new TypeError("family must be a nonempty string");
  }
}

function validateFinite(value: unknown, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number`);
  }
}

function validateOutcome(value: unknown): void {
  if (value !== null && typeof value !== "boolean") {
    throw new TypeError("outcome must be boolean or null");
  }
}

function validateRating(value: unknown): void {
  if (value === null) return;
  validateFinite(value, "rating");
  if (typeof value !== "number" || value < 0 || value > 10) {
    throw new RangeError("rating must be in [0, 10] or null");
  }
}

function validateWinner(value: unknown): void {
  if (value !== null && value !== "A" && value !== "B" && value !== "tie") {
    throw new TypeError("winner must be A, B, tie or null");
  }
}

function validateInteger(value: number, min: number, max: number, label: string): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} must be an integer in [${min}, ${max}]`);
  }
}

function sum(values: ArrayLike<number>): number {
  let total = 0;
  let correction = 0;
  for (let i = 0; i < values.length; i += 1) {
    const adjusted = values[i] - correction;
    const next = total + adjusted;
    correction = (next - total) - adjusted;
    total = next;
  }
  return total;
}

function maxMagnitude(values: readonly number[]): number {
  let scale = 0;
  for (const value of values) scale = Math.max(scale, Math.abs(value));
  return scale;
}

function normalizedMean(values: ArrayLike<number>): number {
  return Math.max(-1, Math.min(1, sum(values) / values.length));
}

function mean(values: readonly number[]): number {
  const scale = maxMagnitude(values);
  return scale === 0 ? 0 : normalizedMean(values.map((value) => value / scale)) * scale;
}

function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Linear interpolation between adjacent order statistics (R type 7). */
function percentile(sorted: readonly number[], probability: number): number {
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[Math.ceil(position)] * weight;
}

function compareText(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

/** Family counts include unknown-only families; empty input is a valid zero-count summary. */
export function summarizeOutcomes(rows: readonly PairedOutcome[]): OutcomeSummary {
  validateArray(rows, "rows");
  const ids = new Set<string>();
  const families = new Set<string>();
  const result: OutcomeSummary = {
    caseCount: rows.length, observedPairs: 0, familyCount: 0,
    fixed: 0, broken: 0, unchangedCorrect: 0, unchangedWrong: 0, unknown: 0,
  };
  for (const row of rows) {
    validateRecord(row, "row");
    validateId(row.id, ids);
    validateFamily(row.family);
    validateOutcome(row.baseline);
    validateOutcome(row.intervention);
    families.add(row.family);
    if (row.baseline === null || row.intervention === null) result.unknown += 1;
    else if (!row.baseline && row.intervention) result.fixed += 1;
    else if (row.baseline && !row.intervention) result.broken += 1;
    else if (row.baseline) result.unchangedCorrect += 1;
    else result.unchangedWrong += 1;
  }
  result.observedPairs = result.caseCount - result.unknown;
  result.familyCount = families.size;
  return result;
}

/**
 * Intervention minus baseline, averaged within families then equally across families.
 * The two-sided sign-flip test includes ties. Exact enumeration uses no +1 correction;
 * Monte Carlo uses (1 + extreme draws) / (1 + draws). Neither treats rows as replicates.
 */
export function familyPairedEffect(
  rows: readonly PairedMetric[],
  options: FamilyPairedEffectOptions = {},
): FamilyPairedEffectResult {
  validateArray(rows, "rows");
  if (rows.length === 0) throw new RangeError("rows must be nonempty");
  validateRecord(options, "options");
  for (const key of Object.keys(options)) {
    if (key !== "seed" && key !== "bootstrapReplicates" && key !== "signFlipReplicates") {
      throw new TypeError(`Unknown option: ${key}`);
    }
  }
  const { seed = 0x5eed1234, bootstrapReplicates = 2000, signFlipReplicates = 10_000 } = options;
  validateInteger(seed, 0, 0xffffffff, "seed");
  validateInteger(bootstrapReplicates, 2, STATISTICS_LIMITS.maxReplicates, "bootstrapReplicates");
  validateInteger(signFlipReplicates, 2, STATISTICS_LIMITS.maxReplicates, "signFlipReplicates");
  const ids = new Set<string>();
  const grouped = new Map<string, { id: string; delta: number }[]>();
  for (const row of rows) {
    validateRecord(row, "row");
    validateId(row.id, ids);
    validateFamily(row.family);
    validateFinite(row.baseline, "baseline");
    validateFinite(row.intervention, "intervention");
    const delta = row.intervention - row.baseline;
    validateFinite(delta, "paired delta");
    const family = grouped.get(row.family) ?? [];
    family.push({ id: row.id, delta });
    grouped.set(row.family, family);
  }
  const familyCount = grouped.size;
  const exact = familyCount <= STATISTICS_LIMITS.exactFamilyLimit;
  const replicates = {
    bootstrap: familyCount < 2 ? 0 : bootstrapReplicates,
    signFlip: exact ? 2 ** familyCount : signFlipReplicates,
  };
  if (familyCount * (replicates.bootstrap + replicates.signFlip) > STATISTICS_LIMITS.maxResamplingOperations) {
    throw new RangeError("Resampling operation budget exceeded; reduce replicates or families");
  }
  const familyEffects = [...grouped.entries()]
    .sort(([first], [second]) => compareText(first, second))
    .map(([family, cases]): FamilyEffect => ({
      family,
      caseCount: cases.length,
      meanDelta: mean(cases.sort((a, b) => compareText(a.id, b.id)).map((row) => row.delta)),
    }));
  const effects = familyEffects.map((family) => family.meanDelta);
  const scale = maxMagnitude(effects);
  const normalized = effects.map((effect) => scale === 0 ? 0 : effect / scale);
  const meanDelta = normalizedMean(normalized) * scale;
  const sample = new Float64Array(familyCount);
  let confidenceInterval95: [number, number] | null = null;
  if (replicates.bootstrap > 0) {
    // Separate streams keep the Monte Carlo p-value independent of bootstrap draw count.
    const random = randomGenerator((seed ^ 0xa5a5a5a5) >>> 0);
    const draws: number[] = [];
    for (let replicate = 0; replicate < replicates.bootstrap; replicate += 1) {
      for (let i = 0; i < familyCount; i += 1) sample[i] = normalized[Math.floor(random() * familyCount)];
      draws.push(normalizedMean(sample));
    }
    draws.sort((a, b) => a - b);
    confidenceInterval95 = [percentile(draws, 0.025) * scale, percentile(draws, 0.975) * scale];
  }
  const observed = Math.abs(sum(normalized));
  // Only accommodate floating-point summation error, on the normalized effect scale.
  const tolerance = 8 * Number.EPSILON * sum(normalized.map(Math.abs));
  const random = randomGenerator((seed ^ 0x5a5a5a5a) >>> 0);
  let extreme = 0;
  for (let replicate = 0; replicate < replicates.signFlip; replicate += 1) {
    for (let i = 0; i < familyCount; i += 1) {
      const positive = exact ? (replicate & (1 << i)) !== 0 : random() >= 0.5;
      sample[i] = positive ? normalized[i] : -normalized[i];
    }
    if (Math.abs(sum(sample)) >= observed - tolerance) extreme += 1;
  }
  return {
    caseCount: rows.length, familyCount, familyEffects, meanDelta, confidenceInterval95,
    pValue: exact ? extreme / replicates.signFlip : (extreme + 1) / (replicates.signFlip + 1),
    exact, lowPower: familyCount < 10, seed, replicates,
    metadata: {
      unit: "family", alternative: "two-sided", confidenceLevel: 0.95, bootstrapMethod: "percentile",
      interpretation: "Equal-weight family effects; bootstrap intervals assume independent, representative families. "
        + "Sign-flip p-values assume independent family effects symmetric about zero under the null. "
        + "Exact enumeration is a symmetry test, not automatically a valid randomization test. "
        + "Few families imply discrete p-values and unstable intervals; rows within a family are not independent replicates.",
    },
  };
}

/** Descriptive controls plus a conservative, tie-inclusive one-sided empirical rank. */
export function randomControlTest(observed: number, controls: readonly number[]): RandomControlResult {
  validateFinite(observed, "observed");
  validateArray(controls, "controls");
  if (controls.length < 2) throw new RangeError("At least two controls are required");
  let min = Infinity;
  let max = -Infinity;
  let rank = 1;
  for (const control of controls) {
    validateFinite(control, "control");
    min = Math.min(min, control);
    max = Math.max(max, control);
    if (control >= observed) rank += 1;
  }
  const scale = maxMagnitude(controls);
  const normalized = controls.map((value) => scale === 0 ? 0 : value / scale);
  const center = normalizedMean(normalized);
  const sd = Math.sqrt(sum(normalized.map((value) => (value - center) ** 2)) / (controls.length - 1)) * scale;
  validateFinite(sd, "control sample standard deviation");
  return {
    n: controls.length, mean: center * scale, sd, rank, pValue: rank / (controls.length + 1), min, max,
    metadata: {
      interpretation: "Descriptive control distribution and one-sided empirical rank, including ties. "
        + "Not parametric significance; inferential use requires exchangeable, appropriately matched random controls.",
    },
  };
}

/** Holm step-down family-wise error adjustment, returned in the original order. */
export function holmAdjust(pValues: readonly number[]): number[] {
  validateArray(pValues, "pValues");
  const sorted = pValues.map((pValue, index) => {
    validateFinite(pValue, "p-value");
    if (pValue < 0 || pValue > 1) throw new RangeError("p-value must be in [0, 1]");
    return { pValue, index };
  }).sort((a, b) => a.pValue - b.pValue);
  const adjusted: number[] = new Array(pValues.length);
  let previous = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    previous = Math.max(previous, Math.min(1, sorted[i].pValue * (sorted.length - i)));
    adjusted[sorted[i].index] = previous;
  }
  return adjusted;
}

/** Scores may be fractional; abstentions are excluded only from the descriptive metrics. */
export function judgeAgreement(pairs: readonly JudgePair[]): JudgeAgreementResult {
  validateArray(pairs, "pairs");
  const ids = new Set<string>();
  let completeCount = 0;
  let firstAbstentionCount = 0;
  let secondAbstentionCount = 0;
  let bothAbstentionCount = 0;
  let exact = 0;
  let withinOne = 0;
  const differences: number[] = [];
  for (const pair of pairs) {
    validateRecord(pair, "pair");
    validateId(pair.id, ids);
    validateRating(pair.first);
    validateRating(pair.second);
    if (pair.first === null) firstAbstentionCount += 1;
    if (pair.second === null) secondAbstentionCount += 1;
    if (pair.first === null && pair.second === null) bothAbstentionCount += 1;
    if (pair.first === null || pair.second === null) continue;
    completeCount += 1;
    const difference = Math.abs(pair.first - pair.second);
    differences.push(difference);
    if (difference === 0) exact += 1;
    if (difference <= 1) withinOne += 1;
  }
  return {
    caseCount: pairs.length, completeCount, abstentionCount: pairs.length - completeCount,
    firstAbstentionCount, secondAbstentionCount, bothAbstentionCount,
    exactFraction: completeCount === 0 ? null : exact / completeCount,
    withinOneFraction: completeCount === 0 ? null : withinOne / completeCount,
    meanAbsoluteDifference: completeCount === 0 ? null : mean(differences),
    metadata: {
      interpretation: "Descriptive paired judge agreement, not independent evidence of correctness or validity. "
        + "Correlated judges can agree while sharing errors; abstention counts include all incomplete pairs.",
    },
  };
}

export function counterbalancedPreference(
  pairs: readonly CounterbalancedPreferencePair[],
): CounterbalancedPreferenceResult {
  validateArray(pairs, "pairs");
  const ids = new Set<string>();
  let completeCount = 0;
  let consistentCount = 0;
  for (const pair of pairs) {
    validateRecord(pair, "pair");
    validateId(pair.id, ids);
    validateWinner(pair.originalWinner);
    validateWinner(pair.reversedWinner);
    if (pair.originalWinner === null || pair.reversedWinner === null) continue;
    completeCount += 1;
    const reversedIdentity = pair.reversedWinner === "A" ? "B" : pair.reversedWinner === "B" ? "A" : "tie";
    if (pair.originalWinner === reversedIdentity) consistentCount += 1;
  }
  return {
    caseCount: pairs.length, completeCount, abstentionCount: pairs.length - completeCount,
    consistentCount, inconsistentCount: completeCount - consistentCount,
    consistencyFraction: completeCount === 0 ? null : consistentCount / completeCount,
    metadata: {
      interpretation: "Reversed positions are mapped to original candidate identities. "
        + "Descriptive consistency only, not a preference ranking or independent evidence.",
    },
  };
}
