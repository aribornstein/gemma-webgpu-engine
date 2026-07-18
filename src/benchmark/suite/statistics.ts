import { createSeededRandom } from "./random";
import type { MetricStatistics } from "./types";

export function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) throw new Error("Percentile requires at least one value");
  if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) {
    throw new Error("Percentile quantile must be between zero and one");
  }
  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function bootstrapMedianConfidenceInterval(
  values: readonly number[],
  options: { seed?: number; resamples?: number } = {},
): { low: number; high: number } {
  if (values.length === 0) throw new Error("Bootstrap confidence interval requires values");
  const resamples = options.resamples ?? 2000;
  if (!Number.isInteger(resamples) || resamples < 100) {
    throw new Error("Bootstrap resamples must be an integer >= 100");
  }
  const random = createSeededRandom(options.seed ?? 0x4e554d41);
  const medians: number[] = [];
  for (let sampleIndex = 0; sampleIndex < resamples; sampleIndex += 1) {
    const sample = Array.from({ length: values.length }, () =>
      values[Math.floor(random() * values.length)]
    );
    medians.push(percentile(sample, 0.5));
  }
  return {
    low: round(percentile(medians, 0.025)),
    high: round(percentile(medians, 0.975)),
  };
}

export function calculateStatistics(
  values: readonly number[],
  options: { seed?: number; bootstrapResamples?: number } = {},
): MetricStatistics | null {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    finite.length;
  const standardDeviation = Math.sqrt(variance);
  return {
    sampleCount: finite.length,
    median: round(percentile(finite, 0.5)),
    mean: round(mean),
    standardDeviation: round(standardDeviation),
    minimum: round(Math.min(...finite)),
    maximum: round(Math.max(...finite)),
    p50: round(percentile(finite, 0.5)),
    p90: round(percentile(finite, 0.9)),
    p95: finite.length >= 20 ? round(percentile(finite, 0.95)) : null,
    p95Status: finite.length >= 20 ? "available" : "insufficient-samples",
    coefficientOfVariation: mean === 0 ? null : round(standardDeviation / Math.abs(mean)),
    medianConfidenceInterval95: bootstrapMedianConfidenceInterval(finite, {
      seed: options.seed,
      resamples: options.bootstrapResamples,
    }),
  };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}