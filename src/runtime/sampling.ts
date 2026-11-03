import type { DecodingConfig } from "./decoding";

export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0 || 0x6d2b79f5;
  }

  next(): number {
    let value = this.state += 0x6d2b79f5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  }
}

interface SamplingCandidate {
  token: number;
  probability: number;
  surprise: number;
}

interface SamplingScratch {
  logits: Float64Array;
  scaled: Float64Array;
  probabilities: Float64Array;
  keysHigh: Uint32Array;
  keysLow: Uint32Array;
  order: Uint32Array;
  orderSwap: Uint32Array;
  candidates: Uint32Array;
}

const RADIX_BUCKETS = 1 << 16;
const radixCounts = new Uint32Array(RADIX_BUCKETS);
const keyFloat = new Float64Array(1);
const keyWords = new Uint32Array(keyFloat.buffer);
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const HIGH_WORD = LITTLE_ENDIAN ? 1 : 0;
const LOW_WORD = LITTLE_ENDIAN ? 0 : 1;
let scratch: SamplingScratch | null = null;

function scratchFor(length: number): SamplingScratch {
  if (scratch?.logits.length !== length) {
    scratch = {
      logits: new Float64Array(length),
      scaled: new Float64Array(length),
      probabilities: new Float64Array(length),
      keysHigh: new Uint32Array(length),
      keysLow: new Uint32Array(length),
      order: new Uint32Array(length),
      orderSwap: new Uint32Array(length),
      candidates: new Uint32Array(length),
    };
  }
  return scratch;
}

/**
 * Applies `cap * tanh(logit / cap)` in f64. Non-finite values (constraint masks) pass through so
 * masked tokens stay impossible. The transform is strictly monotonic, so it never changes an
 * argmax of distinct values.
 */
export function softcapLogit(value: number, cap: number | undefined): number {
  if (cap === undefined || !Number.isFinite(value)) return value;
  return cap * Math.tanh(value / cap);
}

function validateSoftcap(cap: number | undefined): void {
  if (cap !== undefined && (!Number.isFinite(cap) || cap <= 0)) {
    throw new Error("Logit softcap must be a positive finite number");
  }
}

/**
 * Selects the next token. Results are identical to a stable descending sort of every scaled
 * logit followed by top-k, min-p, typical-p, and top-p filtering in that order; the ordering is
 * produced by a stable radix sort and probabilities are accumulated in the same rank order.
 * `logitSoftcap` applies the model's final-logit softcap before penalties and temperature, matching
 * where the reference model applies it (inside the forward pass).
 */
export function sampleToken(
  rawLogits: ArrayLike<number>,
  history: readonly number[],
  config: DecodingConfig,
  random: () => number,
  logitSoftcap?: number,
): number {
  if (rawLogits.length === 0) throw new Error("Cannot sample empty logits");
  validateSoftcap(logitSoftcap);
  const length = rawLogits.length;
  const buffers = scratchFor(length);
  const logits = buffers.logits;
  let hasNaN = false;
  for (let index = 0; index < length; index += 1) {
    const value = rawLogits[index];
    logits[index] = softcapLogit(value, logitSoftcap);
    if (value !== value) hasNaN = true;
  }
  applyHistoryPenalties(logits, length, history, config);
  if (config.temperature === 0) return argmax(logits);

  const scaled = buffers.scaled;
  for (let index = 0; index < length; index += 1) {
    const value = logits[index] / config.temperature;
    scaled[index] = value;
    if (value !== value) hasNaN = true;
  }
  // NaN comparisons make comparator sorting order-dependent; keep the reference behavior.
  if (hasNaN) return sampleSortedCandidates(sortCandidatesByComparator(scaled), config, random);

  const order = descendingStableOrder(scaled, buffers);
  const maximum = scaled[order[0]];
  if (!Number.isFinite(maximum)) {
    return sampleSortedCandidates(sortCandidatesByComparator(scaled), config, random);
  }
  const probabilities = buffers.probabilities;
  let total = 0;
  for (let rank = 0; rank < length; rank += 1) {
    const probability = Math.exp(scaled[order[rank]] - maximum);
    probabilities[rank] = probability;
    total += probability;
  }
  for (let rank = 0; rank < length; rank += 1) probabilities[rank] /= total;

  return selectRankedCandidate(order, probabilities, length, config, random, buffers.candidates);
}

/**
 * Applies top-k, min-p, typical-p, and top-p to rank-ordered tokens and draws one. `order[rank]`
 * is the token at `rank` and `probabilities[rank]` its full-vocabulary probability.
 */
function selectRankedCandidate(
  order: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  length: number,
  config: DecodingConfig,
  random: () => number,
  candidates: Uint32Array,
): number {
  // Candidates are ranks into `order`/`probabilities`, kept in descending-probability order.
  let candidateCount = config.topK > 0 ? Math.min(config.topK, length) : length;
  for (let rank = 0; rank < candidateCount; rank += 1) candidates[rank] = rank;
  if (config.minP > 0) {
    const threshold = probabilities[candidates[0]] * config.minP;
    let kept = 0;
    for (let index = 0; index < candidateCount; index += 1) {
      const rank = candidates[index];
      if (probabilities[rank] >= threshold) candidates[kept++] = rank;
    }
    candidateCount = kept;
  }
  if (config.typicalP < 1) {
    const items: SamplingCandidate[] = [];
    for (let index = 0; index < candidateCount; index += 1) {
      const rank = candidates[index];
      items.push({ token: order[rank], probability: probabilities[rank], surprise: 0 });
    }
    return sampleTypicalCandidates(items, config, random);
  }
  if (config.topP < 1) {
    let cumulative = 0;
    let end = 0;
    do {
      cumulative += probabilities[candidates[end]];
      end += 1;
    } while (end < candidateCount && cumulative < config.topP);
    candidateCount = Math.max(1, end);
  }

  let candidateTotal = 0;
  for (let index = 0; index < candidateCount; index += 1) {
    candidateTotal += probabilities[candidates[index]];
  }
  let cursor = random() * candidateTotal;
  for (let index = 0; index < candidateCount; index += 1) {
    const rank = candidates[index];
    cursor -= probabilities[rank];
    if (cursor <= 0) return order[rank];
  }
  return order[candidates[candidateCount - 1]];
}

/**
 * Samples from GPU-selected top candidates. `tokenIds`/`rawLogits` must be the first ranks of a
 * stable descending order of the raw logits (ties by lower token ID), at least `config.topK` long,
 * and `tailMass` must be `sum(exp(scaled - scaledMax))` over every token ranked after
 * `config.topK`. Softcap and temperature are strictly monotonic, so this is the same candidate
 * set and order `sampleToken` builds; only the softmax normalizer's tail term comes from the
 * caller, which can move a top-p cutoff that sits within that term's rounding error.
 */
export function sampleTopCandidates(
  tokenIds: ArrayLike<number>,
  rawLogits: ArrayLike<number>,
  tailMass: number,
  config: DecodingConfig,
  random: () => number,
  logitSoftcap?: number,
): number {
  validateSoftcap(logitSoftcap);
  if (!(config.temperature > 0)) throw new Error("Top-candidate sampling requires a positive temperature");
  const count = config.topK;
  if (!Number.isInteger(count) || count < 1 || count > tokenIds.length || count > rawLogits.length) {
    throw new Error("Top-candidate sampling needs at least topK ranked candidates");
  }
  if (!Number.isFinite(tailMass) || tailMass < 0) throw new Error("Top-candidate tail mass must be finite");
  const probabilities = new Float64Array(count);
  const maximum = softcapLogit(rawLogits[0], logitSoftcap) / config.temperature;
  if (!Number.isFinite(maximum)) throw new Error("Top-candidate sampling needs a finite maximum");
  let total = 0;
  for (let rank = 0; rank < count; rank += 1) {
    const probability = Math.exp(softcapLogit(rawLogits[rank], logitSoftcap) / config.temperature - maximum);
    probabilities[rank] = probability;
    total += probability;
  }
  total += tailMass;
  for (let rank = 0; rank < count; rank += 1) probabilities[rank] /= total;
  return selectRankedCandidate(tokenIds, probabilities, count, config, random, new Uint32Array(count));
}

export function tokenLogProbability(
  rawLogits: ArrayLike<number>,
  token: number,
  logitSoftcap?: number,
): number {
  if (!Number.isInteger(token) || token < 0 || token >= rawLogits.length) {
    throw new Error(`Token ${token} is outside logits length ${rawLogits.length}`);
  }
  validateSoftcap(logitSoftcap);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < rawLogits.length; index += 1) {
    maximum = Math.max(maximum, softcapLogit(rawLogits[index], logitSoftcap));
  }
  let denominator = 0;
  for (let index = 0; index < rawLogits.length; index += 1) {
    denominator += Math.exp(softcapLogit(rawLogits[index], logitSoftcap) - maximum);
  }
  return softcapLogit(rawLogits[token], logitSoftcap) - maximum - Math.log(denominator);
}

function sortCandidatesByComparator(scaledLogits: Float64Array): SamplingCandidate[] {
  const scaled = Array.from(scaledLogits, (logit, token) => ({ token, logit, probability: 0, surprise: 0 }))
    .sort((left, right) => right.logit - left.logit);
  const maximum = scaled[0].logit;
  let total = 0;
  for (const item of scaled) {
    item.probability = Math.exp(item.logit - maximum);
    total += item.probability;
  }
  for (const item of scaled) item.probability /= total;
  return scaled;
}

function sampleSortedCandidates(
  scaled: SamplingCandidate[],
  config: DecodingConfig,
  random: () => number,
): number {
  let candidates = config.topK > 0 ? scaled.slice(0, config.topK) : scaled;
  if (config.minP > 0) {
    const threshold = candidates[0].probability * config.minP;
    candidates = candidates.filter((item) => item.probability >= threshold);
  }
  return sampleTypicalCandidates(candidates, config, random);
}

function sampleTypicalCandidates(
  initial: SamplingCandidate[],
  config: DecodingConfig,
  random: () => number,
): number {
  let candidates = initial;
  if (config.typicalP < 1) {
    const entropy = candidates.reduce((sum, item) => sum - item.probability * Math.log(item.probability), 0);
    for (const item of candidates) item.surprise = Math.abs(-Math.log(item.probability) - entropy);
    candidates.sort((left, right) => left.surprise - right.surprise);
    candidates = cumulativeCut(candidates, config.typicalP);
    candidates.sort((left, right) => right.probability - left.probability);
  }
  if (config.topP < 1) candidates = cumulativeCut(candidates, config.topP);

  const candidateTotal = candidates.reduce((sum, item) => sum + item.probability, 0);
  let cursor = random() * candidateTotal;
  for (const item of candidates) {
    cursor -= item.probability;
    if (cursor <= 0) return item.token;
  }
  return candidates[candidates.length - 1].token;
}

function cumulativeCut<T extends { probability: number }>(items: T[], threshold: number): T[] {
  let cumulative = 0;
  let end = 0;
  do {
    cumulative += items[end].probability;
    end += 1;
  } while (end < items.length && cumulative < threshold);
  return items.slice(0, Math.max(1, end));
}

/**
 * Returns token indices ordered by descending value, ties in ascending index order. Values must
 * not be NaN. Signed zeros compare equal, matching `right - left` comparator semantics.
 */
function descendingStableOrder(values: Float64Array, buffers: SamplingScratch): Uint32Array {
  const length = values.length;
  const { keysHigh, keysLow } = buffers;
  for (let index = 0; index < length; index += 1) {
    const value = values[index];
    keyFloat[0] = value === 0 ? 0 : value;
    let high = keyWords[HIGH_WORD];
    let low = keyWords[LOW_WORD];
    if (high & 0x80000000) {
      high = ~high >>> 0;
      low = ~low >>> 0;
    } else {
      high = (high | 0x80000000) >>> 0;
    }
    keysHigh[index] = ~high >>> 0;
    keysLow[index] = ~low >>> 0;
  }
  let order = buffers.order;
  let swap = buffers.orderSwap;
  for (let index = 0; index < length; index += 1) order[index] = index;
  const passes: readonly [Uint32Array, number][] = [
    [keysLow, 0], [keysLow, 16], [keysHigh, 0], [keysHigh, 16],
  ];
  for (const [keys, shift] of passes) {
    radixCounts.fill(0);
    for (let index = 0; index < length; index += 1) {
      radixCounts[(keys[order[index]] >>> shift) & 0xffff] += 1;
    }
    let sum = 0;
    for (let bucket = 0; bucket < RADIX_BUCKETS; bucket += 1) {
      const count = radixCounts[bucket];
      radixCounts[bucket] = sum;
      sum += count;
    }
    for (let index = 0; index < length; index += 1) {
      const token = order[index];
      swap[radixCounts[(keys[token] >>> shift) & 0xffff]++] = token;
    }
    const previous = order;
    order = swap;
    swap = previous;
  }
  // After an even number of passes the result is back in `buffers.order`.
  return order;
}

function applyHistoryPenalties(
  logits: Float64Array,
  length: number,
  history: readonly number[],
  config: DecodingConfig,
): void {
  const counts = new Map<number, number>();
  const recentHistory = config.repetitionWindow === 0
    ? []
    : history.slice(-config.repetitionWindow);
  for (const token of recentHistory) counts.set(token, (counts.get(token) ?? 0) + 1);
  for (const [token, count] of counts) {
    if (token < 0 || token >= length) continue;
    logits[token] = logits[token] < 0 ? logits[token] * config.repetitionPenalty : logits[token] / config.repetitionPenalty;
    logits[token] -= config.presencePenalty + count * config.frequencyPenalty;
  }
}

/**
 * Log-softmax of the logits after the softcap and history penalties that `sampleToken` applies.
 * Temperature and truncation filters are not applied; beam search scores the full distribution.
 * The returned array is owned by the caller.
 */
export function processedLogProbabilities(
  rawLogits: ArrayLike<number>,
  history: readonly number[],
  config: DecodingConfig,
  logitSoftcap?: number,
): Float64Array {
  validateSoftcap(logitSoftcap);
  const length = rawLogits.length;
  const values = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = softcapLogit(rawLogits[index], logitSoftcap);
  }
  applyHistoryPenalties(values, length, history, config);
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < length; index += 1) {
    if (values[index] > maximum) maximum = values[index];
  }
  if (!Number.isFinite(maximum)) throw new Error("Cannot normalize logits without a finite maximum");
  let denominator = 0;
  for (let index = 0; index < length; index += 1) denominator += Math.exp(values[index] - maximum);
  const normalizer = maximum + Math.log(denominator);
  for (let index = 0; index < length; index += 1) values[index] -= normalizer;
  return values;
}

function argmax(values: Float64Array): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}
