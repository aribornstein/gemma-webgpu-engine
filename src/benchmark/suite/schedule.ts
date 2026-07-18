import { createSeededRandom, seededShuffle } from "./random";
import type {
  BenchmarkCase,
  BenchmarkMode,
  BenchmarkScheduleEntry,
} from "./types";

export interface ScheduleOptions {
  seed: number;
  modes: readonly BenchmarkMode[];
  workloads: readonly BenchmarkCase[];
  runtimeIds: readonly string[];
  iterations: number;
  includeConversationVariants?: boolean;
}

export function createBenchmarkSchedule(options: ScheduleOptions): BenchmarkScheduleEntry[] {
  if (!Number.isInteger(options.iterations) || options.iterations < 1) {
    throw new Error("Benchmark schedule iterations must be an integer >= 1");
  }
  const random = createSeededRandom(options.seed);
  const schedule: BenchmarkScheduleEntry[] = [];
  let sequence = 0;
  let block = 0;
  for (const mode of options.modes) {
    for (const workload of options.workloads) {
      for (let iteration = 0; iteration < options.iterations; iteration += 1) {
        const variants = mode === "conversation-cache" && options.includeConversationVariants !== false
          ? ["reused", "fresh"] as const
          : ["not-applicable"] as const;
        for (const conversationVariant of variants) {
          const randomizedRuntimes = seededShuffle(options.runtimeIds, random);
          for (const runtimeId of randomizedRuntimes) {
            schedule.push({
              sequence,
              block,
              mode,
              workloadId: workload.id,
              iteration,
              runtimeId,
              conversationVariant,
            });
            sequence += 1;
          }
          block += 1;
        }
      }
    }
  }
  return schedule;
}