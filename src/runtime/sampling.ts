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

export function sampleToken(rawLogits: ArrayLike<number>, history: readonly number[], config: DecodingConfig, random: () => number): number {
  if (rawLogits.length === 0) throw new Error("Cannot sample empty logits");
  const logits = Array.from(rawLogits);
  const counts = new Map<number, number>();
  const recentHistory = config.repetitionWindow === 0
    ? []
    : history.slice(-config.repetitionWindow);
  for (const token of recentHistory) counts.set(token, (counts.get(token) ?? 0) + 1);
  for (const [token, count] of counts) {
    if (token < 0 || token >= logits.length) continue;
    logits[token] = logits[token] < 0 ? logits[token] * config.repetitionPenalty : logits[token] / config.repetitionPenalty;
    logits[token] -= config.presencePenalty + count * config.frequencyPenalty;
  }
  if (config.temperature === 0) return argmax(logits);

  const scaled = logits
    .map((value, token) => ({ token, logit: value / config.temperature, probability: 0, surprise: 0 }))
    .sort((left, right) => right.logit - left.logit);
  const maximum = scaled[0].logit;
  let total = 0;
  for (const item of scaled) {
    item.probability = Math.exp(item.logit - maximum);
    total += item.probability;
  }
  for (const item of scaled) item.probability /= total;

  let candidates = config.topK > 0 ? scaled.slice(0, config.topK) : scaled;
  if (config.minP > 0) {
    const threshold = candidates[0].probability * config.minP;
    candidates = candidates.filter((item) => item.probability >= threshold);
  }
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

function argmax(values: readonly number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}
