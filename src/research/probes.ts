/**
 * Low-capacity linear probes for research captures: train-split standardization, difference of
 * class means, and L2-regularized logistic regression (full-batch gradient descent on the probe
 * only), with AUC and label-shuffled controls. Deterministic for a given seed.
 */
export interface ProbeData {
  x: Float32Array[];
  y: number[];
  group: string[];
}

export interface FittedProbe {
  kind: "mean_difference" | "logistic";
  mean: Float64Array;
  scale: Float64Array;
  weights: Float64Array;
  bias: number;
}

export function standardizer(x: readonly Float32Array[]): { mean: Float64Array; scale: Float64Array } {
  const dim = x[0].length;
  const mean = new Float64Array(dim);
  const scale = new Float64Array(dim);
  for (const row of x) for (let i = 0; i < dim; i += 1) mean[i] += row[i] / x.length;
  for (const row of x) for (let i = 0; i < dim; i += 1) scale[i] += (row[i] - mean[i]) ** 2 / x.length;
  for (let i = 0; i < dim; i += 1) scale[i] = Math.sqrt(scale[i]) || 1;
  return { mean, scale };
}

function standardized(row: Float32Array, mean: Float64Array, scale: Float64Array): Float64Array {
  const out = new Float64Array(row.length);
  for (let i = 0; i < row.length; i += 1) out[i] = (row[i] - mean[i]) / scale[i];
  return out;
}

export function fitMeanDifference(data: ProbeData): FittedProbe {
  const { mean, scale } = standardizer(data.x);
  const dim = mean.length;
  const positive = new Float64Array(dim);
  const negative = new Float64Array(dim);
  let positives = 0;
  let negatives = 0;
  data.x.forEach((row, index) => {
    const z = standardized(row, mean, scale);
    const target = data.y[index] === 1 ? positive : negative;
    for (let i = 0; i < dim; i += 1) target[i] += z[i];
    if (data.y[index] === 1) positives += 1; else negatives += 1;
  });
  if (!positives || !negatives) throw new Error("Mean-difference probe needs both classes");
  const weights = new Float64Array(dim);
  for (let i = 0; i < dim; i += 1) weights[i] = positive[i] / positives - negative[i] / negatives;
  return { kind: "mean_difference", mean, scale, weights, bias: 0 };
}

export function fitLogistic(data: ProbeData, l2 = 1, iterations = 300, learningRate = 0.5): FittedProbe {
  const { mean, scale } = standardizer(data.x);
  const rows = data.x.map((row) => standardized(row, mean, scale));
  const dim = mean.length;
  const weights = new Float64Array(dim);
  let bias = 0;
  const n = rows.length;
  for (let step = 0; step < iterations; step += 1) {
    const gradient = new Float64Array(dim);
    let biasGradient = 0;
    for (let r = 0; r < n; r += 1) {
      let logit = bias;
      for (let i = 0; i < dim; i += 1) logit += weights[i] * rows[r][i];
      const error = 1 / (1 + Math.exp(-logit)) - data.y[r];
      for (let i = 0; i < dim; i += 1) gradient[i] += error * rows[r][i] / n;
      biasGradient += error / n;
    }
    for (let i = 0; i < dim; i += 1) weights[i] -= learningRate * (gradient[i] + l2 * weights[i] / n);
    bias -= learningRate * biasGradient;
  }
  return { kind: "logistic", mean, scale, weights, bias };
}

export function probeScore(probe: FittedProbe, row: Float32Array): number {
  let total = probe.bias;
  for (let i = 0; i < row.length; i += 1) total += probe.weights[i] * (row[i] - probe.mean[i]) / probe.scale[i];
  return total;
}

/** Probe direction in raw activation space (weights divided by the standardization scale). */
export function rawDirection(probe: FittedProbe): Float32Array {
  const out = new Float32Array(probe.weights.length);
  for (let i = 0; i < out.length; i += 1) out[i] = probe.weights[i] / probe.scale[i];
  return out;
}

export function auc(scores: readonly number[], labels: readonly number[]): number {
  const pairs = scores.map((score, index) => ({ score, label: labels[index] }));
  const positives = pairs.filter((pair) => pair.label === 1);
  const negatives = pairs.filter((pair) => pair.label === 0);
  if (!positives.length || !negatives.length) return Number.NaN;
  let wins = 0;
  for (const p of positives) for (const q of negatives) wins += p.score > q.score ? 1 : p.score === q.score ? 0.5 : 0;
  return wins / (positives.length * negatives.length);
}

/** Deterministic permutation of labels within the training split (shuffled-label control). */
export function shuffledLabels(labels: readonly number[], seed: number): number[] {
  const out = [...labels];
  let state = seed >>> 0 || 1;
  for (let i = out.length - 1; i > 0; i -= 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Percentile bootstrap CI of an AUC by resampling groups (scenario families). */
export function groupBootstrapAuc(
  scores: readonly number[],
  labels: readonly number[],
  groups: readonly string[],
  replicates = 1000,
  seed = 7,
): [number, number] {
  const unique = [...new Set(groups)];
  let state = seed;
  const next = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;
  const values: number[] = [];
  for (let r = 0; r < replicates; r += 1) {
    const s: number[] = [];
    const l: number[] = [];
    for (let k = 0; k < unique.length; k += 1) {
      const pick = unique[Math.floor(next() * unique.length)];
      groups.forEach((group, index) => {
        if (group === pick) { s.push(scores[index]); l.push(labels[index]); }
      });
    }
    const value = auc(s, l);
    if (Number.isFinite(value)) values.push(value);
  }
  values.sort((a, b) => a - b);
  return [values[Math.floor(0.025 * values.length)], values[Math.floor(0.975 * values.length)]];
}
