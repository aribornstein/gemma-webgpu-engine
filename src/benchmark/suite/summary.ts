import { calculateStatistics } from "./statistics";
import type {
  ArtifactTrack,
  BenchmarkEnvironment,
  BenchmarkScheduleEntry,
  BenchmarkSummary,
  RawBenchmarkRun,
  SummaryRow,
} from "./types";

export function summarizeBenchmarkRuns(
  runs: readonly RawBenchmarkRun[],
  environment: BenchmarkEnvironment,
  schedule: readonly BenchmarkScheduleEntry[],
  limitations: readonly string[] = [],
): BenchmarkSummary {
  const groups = new Map<string, { track: ArtifactTrack; runs: RawBenchmarkRun[] }>();
  for (const run of runs) {
    if (run.browserMode !== environment.browserMode) {
      throw new Error("Cannot aggregate headed and headless benchmark runs");
    }
    for (const track of run.trackEligibility) {
      const key = [
        track,
        run.browserMode,
        run.mode,
        run.schedule.conversationVariant,
        run.runtime.id,
        run.workload.id,
      ].join("\u0000");
      const group = groups.get(key) ?? { track, runs: [] };
      group.runs.push(run);
      groups.set(key, group);
    }
  }
  const rows = [...groups.values()].map(({ track, runs: groupedRuns }, index) =>
    summarizeGroup(track, groupedRuns, environment.benchmarkSeed + index)
  ).sort(compareRows);
  return {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    environment,
    schedule: Object.freeze([...schedule]),
    rows: Object.freeze(rows),
    limitations: Object.freeze([...new Set(limitations)]),
  };
}

function summarizeGroup(track: ArtifactTrack, runs: RawBenchmarkRun[], seed: number): SummaryRow {
  const first = runs[0];
  const valid = runs.filter((run) => run.equalWorkEligible && !run.correctness.invalidOutput);
  const chunkIntervals = valid.flatMap((run) => run.external.streamChunkIntervalMs);
  const stats = (values: number[]) => calculateStatistics(values, { seed });
  return {
    track,
    browserMode: first.browserMode,
    mode: first.mode,
    conversationVariant: first.schedule.conversationVariant,
    runtimeId: first.runtime.id,
    runtimeName: first.runtime.name,
    workloadId: first.workload.id,
    targetInputTokens: first.workload.targetInputTokens,
    targetOutputTokens: first.workload.targetOutputTokens,
    validRuns: valid.length,
    excludedRuns: runs.length - valid.length,
    actualInputTokens: stats(valid.map((run) => run.actualInputTokens)),
    actualOutputTokens: stats(valid.map((run) => run.actualOutputTokens)),
    ttftMs: stats(valid.flatMap((run) => run.external.ttftMs === null ? [] : [run.external.ttftMs])),
    totalMs: stats(valid.map((run) => run.external.totalMs)),
    aggregateDecodeTokensPerSecond: stats(valid.flatMap((run) =>
      run.external.aggregateDecodeTokensPerSecond === null
        ? []
        : [run.external.aggregateDecodeTokensPerSecond]
    )),
    charactersPerSecond: stats(valid.map((run) => run.external.charactersPerSecond)),
    streamChunkIntervalMs: stats(chunkIntervals),
    startupReadyMs: stats(valid.flatMap((run) => run.startup ? [run.startup.readyMs] : [])),
  };
}

function compareRows(left: SummaryRow, right: SummaryRow): number {
  return left.track.localeCompare(right.track) ||
    left.browserMode.localeCompare(right.browserMode) ||
    left.mode.localeCompare(right.mode) ||
    left.conversationVariant.localeCompare(right.conversationVariant) ||
    left.workloadId.localeCompare(right.workloadId) ||
    left.runtimeId.localeCompare(right.runtimeId);
}