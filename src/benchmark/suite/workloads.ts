import type { BenchmarkAdapter, BenchmarkCase } from "./types";

const MATRIX = [
  [32, 32],
  [32, 128],
  [32, 512],
  [153, 128],
  [256, 128],
  [639, 128],
  [1024, 128],
  [4096, 128],
  [8192, 128],
] as const;

export function createBenchmarkWorkloads(): readonly BenchmarkCase[] {
  return Object.freeze(MATRIX.map(([inputTokens, outputTokens]) => ({
    id: `input-${inputTokens}-output-${outputTokens}`,
    targetInputTokens: inputTokens,
    targetOutputTokens: outputTokens,
    prompt: deterministicPrompt(inputTokens, outputTokens),
    expectedPrefix: "alpha beta gamma delta",
    supportsLongContext: inputTokens <= 4096,
  })));
}

export function createSmokeWorkload(): BenchmarkCase {
  return {
    id: "smoke-input-32-output-8",
    targetInputTokens: 32,
    targetOutputTokens: 8,
    prompt: deterministicPrompt(32, 8),
    expectedPrefix: "alpha beta gamma delta",
    supportsLongContext: true,
  };
}

export async function calibrateWorkloadForRuntime(
  adapter: BenchmarkAdapter,
  workload: BenchmarkCase,
): Promise<BenchmarkCase> {
  const prompt = await findClosestPrompt(adapter, workload.targetInputTokens, workload.targetOutputTokens);
  return { ...workload, prompt };
}

function deterministicPrompt(targetInputTokens: number, targetOutputTokens: number): string {
  return promptWithPadding(Math.max(0, targetInputTokens - 20), targetOutputTokens);
}

async function findClosestPrompt(
  adapter: BenchmarkAdapter,
  targetInputTokens: number,
  targetOutputTokens: number,
): Promise<string> {
  let low = 0;
  let high = targetInputTokens * 2;
  let closest = promptWithPadding(0, targetOutputTokens);
  let closestDistance = Number.POSITIVE_INFINITY;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const candidate = promptWithPadding(middle, targetOutputTokens);
    const count = await adapter.countTokens(candidate, "input");
    const distance = Math.abs(count - targetInputTokens);
    if (distance < closestDistance) {
      closest = candidate;
      closestDistance = distance;
    }
    if (count === targetInputTokens) return candidate;
    if (count < targetInputTokens) low = middle + 1;
    else high = middle - 1;
  }
  return closest;
}

function promptWithPadding(paddingWordCount: number, targetOutputTokens: number): string {
  const instruction =
    `Write ${targetOutputTokens} tokens. Repeat: alpha beta gamma delta. Start with alpha. No explanation.`;
  const padding = Array.from({ length: paddingWordCount }, (_, index) =>
    `context${index % 16}`
  ).join(" ");
  return padding.length === 0 ? instruction : `Context: ${padding}\n${instruction}`;
}