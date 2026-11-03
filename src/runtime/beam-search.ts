export interface BeamSearchOptions {
  /** Beams kept per step. 1 is greedy search over the processed distribution. */
  numBeams: number;
  /**
   * Exponent applied to the generated length when ranking finished hypotheses
   * (`sumLogProbability / length ** lengthPenalty`). Values above 0 favor longer outputs.
   */
  lengthPenalty: number;
  /**
   * When true, stop as soon as `numBeams` hypotheses have finished. When false, continue until no
   * running beam can outscore the worst kept hypothesis (the Hugging Face `early_stopping=False`
   * heuristic).
   */
  earlyStopping: boolean;
  /**
   * When above 0, a beam may not generate any n-gram of this many tokens twice (the Hugging Face
   * `no_repeat_ngram_size` rule, applied to generated tokens only). This blocks the repetition
   * loops beam search is prone to. 0 disables it.
   */
  noRepeatNgramSize: number;
}

export const MAX_BEAM_SEARCH_BEAMS = 16;

export const DEFAULT_BEAM_SEARCH_OPTIONS: BeamSearchOptions = {
  numBeams: 4,
  lengthPenalty: 1,
  earlyStopping: false,
  noRepeatNgramSize: 0,
};

export interface BeamHypothesis {
  tokenIds: readonly number[];
  tokenLogProbabilities: readonly number[];
  sumLogProbability: number;
  /** Length-normalized score used for ranking. */
  score: number;
  /** End or stop token that finished the hypothesis, or null when it reached the length limit. */
  terminalTokenId: number | null;
}

export interface BeamSearchStep {
  step: number;
  maxSteps: number;
  runningBeams: readonly { tokenIds: readonly number[]; sumLogProbability: number }[];
  finishedCount: number;
}

export interface BeamSearchDriver {
  readonly options: BeamSearchOptions;
  readonly maxNewTokens: number;
  /** Logits for the first generated position, shared by every beam. */
  readonly initialLogits: ArrayLike<number>;
  isTerminal(tokenId: number): boolean;
  /** Log-probabilities for a beam whose generated tokens are `generatedTokenIds`. */
  logProbabilities(logits: ArrayLike<number>, generatedTokenIds: readonly number[]): ArrayLike<number>;
  /**
   * Makes beam `index` the continuation of previous beam `parents[index]` extended by
   * `tokenIds[index]`, evaluates that token, and returns next-position logits per beam.
   */
  advance(parents: readonly number[], tokenIds: readonly number[]): Promise<readonly ArrayLike<number>[]>;
  onStep?(step: BeamSearchStep): void | Promise<void>;
  throwIfAborted?(): void;
}

export interface BeamSearchResult {
  best: BeamHypothesis;
  /** Kept hypotheses, best first. */
  hypotheses: readonly BeamHypothesis[];
  steps: number;
}

interface RunningBeam {
  tokenIds: number[];
  tokenLogProbabilities: number[];
  sumLogProbability: number;
}

interface Candidate {
  parent: number;
  tokenId: number;
  logProbability: number;
  total: number;
}

export function validateBeamSearchOptions(input: Partial<BeamSearchOptions>): string[] {
  const value = { ...DEFAULT_BEAM_SEARCH_OPTIONS, ...input };
  const errors: string[] = [];
  if (!Number.isInteger(value.numBeams) || value.numBeams < 1 || value.numBeams > MAX_BEAM_SEARCH_BEAMS) {
    errors.push(`numBeams must be an integer from 1 to ${MAX_BEAM_SEARCH_BEAMS}`);
  }
  if (!Number.isFinite(value.lengthPenalty)) errors.push("lengthPenalty must be finite");
  if (typeof value.earlyStopping !== "boolean") errors.push("earlyStopping must be a boolean");
  if (!Number.isInteger(value.noRepeatNgramSize) || value.noRepeatNgramSize < 0) {
    errors.push("noRepeatNgramSize must be a non-negative integer");
  }
  return errors;
}

export function resolveBeamSearchOptions(input: Partial<BeamSearchOptions>): BeamSearchOptions {
  const errors = validateBeamSearchOptions(input);
  if (errors.length > 0) throw new Error(`Invalid beam search options: ${errors.join("; ")}`);
  return { ...DEFAULT_BEAM_SEARCH_OPTIONS, ...input };
}

/** Length-normalized hypothesis score; terminal tokens count toward the length. */
export function beamHypothesisScore(
  sumLogProbability: number,
  length: number,
  lengthPenalty: number,
): number {
  return sumLogProbability / Math.max(1, length) ** lengthPenalty;
}

/**
 * Tokens that would complete an n-gram of `size` tokens already present in `tokenIds`.
 */
export function bannedNgramTokens(tokenIds: readonly number[], size: number): Set<number> {
  const banned = new Set<number>();
  if (size < 1 || tokenIds.length < size - 1) return banned;
  if (size === 1) return new Set(tokenIds);
  const prefixStart = tokenIds.length - (size - 1);
  for (let start = 0; start + size <= tokenIds.length; start += 1) {
    let matches = true;
    for (let offset = 0; offset < size - 1; offset += 1) {
      if (tokenIds[start + offset] !== tokenIds[prefixStart + offset]) {
        matches = false;
        break;
      }
    }
    if (matches) banned.add(tokenIds[start + size - 1]);
  }
  return banned;
}

/**
 * Indices of the `count` largest finite values, largest first; ties keep the lower index first.
 */
export function topLogProbabilityIndices(values: ArrayLike<number>, count: number): number[] {
  const indices: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!(value > Number.NEGATIVE_INFINITY)) continue;
    if (indices.length === count && value <= values[indices[count - 1]]) continue;
    let insert = indices.length;
    while (insert > 0 && values[indices[insert - 1]] < value) insert -= 1;
    indices.splice(insert, 0, index);
    if (indices.length > count) indices.pop();
  }
  return indices;
}

class FinishedHypotheses {
  readonly items: BeamHypothesis[] = [];
  private readonly capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get full(): boolean {
    return this.items.length >= this.capacity;
  }

  get worstScore(): number {
    return this.items.length === 0 ? Number.NEGATIVE_INFINITY : this.items[this.items.length - 1].score;
  }

  add(hypothesis: BeamHypothesis): void {
    if (this.full && hypothesis.score <= this.worstScore) return;
    let insert = this.items.length;
    while (insert > 0 && this.items[insert - 1].score < hypothesis.score) insert -= 1;
    this.items.splice(insert, 0, hypothesis);
    if (this.items.length > this.capacity) this.items.pop();
  }
}

/**
 * Deterministic beam search. Each step expands every running beam by its `2 * numBeams` most
 * likely tokens, ranks the candidates by cumulative log-probability (ties: lower parent, then
 * lower token ID), and keeps the first `numBeams` non-terminal candidates. A terminal candidate
 * ranked inside the first `numBeams` finishes a hypothesis. Hypotheses are ranked by
 * `beamHypothesisScore`, and running beams are finalized when the length limit is reached.
 */
export async function runBeamSearch(driver: BeamSearchDriver): Promise<BeamSearchResult> {
  const { numBeams, lengthPenalty, earlyStopping, noRepeatNgramSize } =
    resolveBeamSearchOptions(driver.options);
  if (!Number.isInteger(driver.maxNewTokens) || driver.maxNewTokens < 1) {
    throw new Error("Beam search maxNewTokens must be a positive integer");
  }
  const finished = new FinishedHypotheses(numBeams);
  let beams: RunningBeam[] = [{ tokenIds: [], tokenLogProbabilities: [], sumLogProbability: 0 }];
  let logits: readonly ArrayLike<number>[] = [driver.initialLogits];
  const candidatesPerBeam = 2 * numBeams;
  let steps = 0;
  for (let step = 0; step < driver.maxNewTokens; step += 1) {
    driver.throwIfAborted?.();
    steps = step + 1;
    const candidates: Candidate[] = [];
    for (let parent = 0; parent < beams.length; parent += 1) {
      const beam = beams[parent];
      let logProbabilities = driver.logProbabilities(logits[parent], beam.tokenIds);
      const banned = bannedNgramTokens(beam.tokenIds, noRepeatNgramSize);
      if (banned.size > 0) {
        const masked = Float64Array.from(logProbabilities);
        for (const tokenId of banned) {
          if (tokenId >= 0 && tokenId < masked.length) masked[tokenId] = Number.NEGATIVE_INFINITY;
        }
        logProbabilities = masked;
      }
      for (const tokenId of topLogProbabilityIndices(logProbabilities, candidatesPerBeam)) {
        const logProbability = logProbabilities[tokenId];
        candidates.push({
          parent,
          tokenId,
          logProbability,
          total: beam.sumLogProbability + logProbability,
        });
      }
    }
    candidates.sort((left, right) =>
      right.total - left.total || left.parent - right.parent || left.tokenId - right.tokenId);

    const next: RunningBeam[] = [];
    const parents: number[] = [];
    for (let rank = 0; rank < candidates.length && next.length < numBeams; rank += 1) {
      const candidate = candidates[rank];
      const parent = beams[candidate.parent];
      if (driver.isTerminal(candidate.tokenId)) {
        if (rank >= numBeams) continue;
        finished.add({
          tokenIds: [...parent.tokenIds],
          tokenLogProbabilities: [...parent.tokenLogProbabilities, candidate.logProbability],
          sumLogProbability: candidate.total,
          score: beamHypothesisScore(candidate.total, parent.tokenIds.length + 1, lengthPenalty),
          terminalTokenId: candidate.tokenId,
        });
        continue;
      }
      next.push({
        tokenIds: [...parent.tokenIds, candidate.tokenId],
        tokenLogProbabilities: [...parent.tokenLogProbabilities, candidate.logProbability],
        sumLogProbability: candidate.total,
      });
      parents.push(candidate.parent);
    }

    await driver.onStep?.({
      step,
      maxSteps: driver.maxNewTokens,
      runningBeams: next.map(({ tokenIds, sumLogProbability }) => ({ tokenIds, sumLogProbability })),
      finishedCount: finished.items.length,
    });

    if (next.length === 0) return resultFrom(finished, steps);
    if (finished.full) {
      if (earlyStopping) return resultFrom(finished, steps);
      const bestRunning = beamHypothesisScore(next[0].sumLogProbability, step + 1, lengthPenalty);
      if (finished.worstScore >= bestRunning) return resultFrom(finished, steps);
    }
    beams = next;
    if (step + 1 === driver.maxNewTokens) break;
    logits = await driver.advance(parents, next.map(({ tokenIds }) => tokenIds[tokenIds.length - 1]));
    if (logits.length !== beams.length) throw new Error("Beam search advance returned the wrong beam count");
  }
  for (const beam of beams) {
    finished.add({
      tokenIds: beam.tokenIds,
      tokenLogProbabilities: beam.tokenLogProbabilities,
      sumLogProbability: beam.sumLogProbability,
      score: beamHypothesisScore(beam.sumLogProbability, beam.tokenIds.length, lengthPenalty),
      terminalTokenId: null,
    });
  }
  return resultFrom(finished, steps);
}

function resultFrom(finished: FinishedHypotheses, steps: number): BeamSearchResult {
  const best = finished.items[0];
  if (!best) throw new Error("Beam search produced no hypothesis");
  return { best, hypotheses: Object.freeze([...finished.items]), steps };
}
