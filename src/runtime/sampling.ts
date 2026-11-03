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
 * Selects the next token. Results are identical to a stable descending sort of every scaled
 * logit followed by top-k, min-p, typical-p, and top-p filtering in that order; the ordering is
 * produced by a stable radix sort and probabilities are accumulated in the same rank order.
 */
export function sampleToken(rawLogits: ArrayLike<number>, history: readonly number[], config: DecodingConfig, random: () => number): number {
  if (rawLogits.length === 0) throw new Error("Cannot sample empty logits");
  const length = rawLogits.length;
  const buffers = scratchFor(length);
  const logits = buffers.logits;
  let hasNaN = false;
  for (let index = 0; index < length; index += 1) {
    const value = rawLogits[index];
    logits[index] = value;
    if (value !== value) hasNaN = true;
  }
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

  // Candidates are ranks into `order`/`probabilities`, kept in descending-probability order.
  const candidates = buffers.candidates;
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

export function tokenLogProbability(rawLogits: ArrayLike<number>, token: number): number {
  if (!Number.isInteger(token) || token < 0 || token >= rawLogits.length) {
    throw new Error(`Token ${token} is outside logits length ${rawLogits.length}`);
  }
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < rawLogits.length; index += 1) {
    maximum = Math.max(maximum, rawLogits[index]);
  }
  let denominator = 0;
  for (let index = 0; index < rawLogits.length; index += 1) {
    denominator += Math.exp(rawLogits[index] - maximum);
  }
  return rawLogits[token] - maximum - Math.log(denominator);
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

function argmax(values: Float64Array): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}
