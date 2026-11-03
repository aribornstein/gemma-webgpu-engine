import {
  loadGemmaGenerationSession,
  type GemmaGenerationSession,
  type GemmaMeasuredGenerationOptions,
  type GemmaSessionLoadOptions,
} from "../runtime/gemma-session";

/**
 * Owned-engine decoding-policy benchmark. The cross-runtime suite is greedy-only because the
 * pinned comparison runtime cannot sample, so it never exercises CPU-side selection or cold
 * token-input loads. These workloads cover both through the public session API.
 */
export type DecodePolicyWorkloadId =
  | "greedy-warm"
  | "sampled-warm"
  | "json-schema-warm"
  | "greedy-fresh-session";

export interface DecodePolicyDistribution {
  medianMs: number;
  p95Ms: number;
  samples: number;
}

export interface DecodePolicyWorkloadResult {
  id: DecodePolicyWorkloadId;
  iterations: number;
  generatedTokens: readonly number[];
  timeToFirstToken: DecodePolicyDistribution;
  interTokenLatency: DecodePolicyDistribution;
}

export interface DecodePolicyBenchmarkArtifact {
  schemaVersion: 1;
  capturedAt: string;
  environment: { userAgent: string; adapterInfo: Record<string, string> };
  configuration: { cacheCapacity: number; iterations: number; maxNewTokens: number };
  workloads: DecodePolicyWorkloadResult[];
}

export interface DecodePolicyBenchmarkOptions extends GemmaSessionLoadOptions {
  iterations?: number;
  maxNewTokens?: number;
}

const PROMPT = "Write a short paragraph about the history of mathematics.";
const JSON_PROMPT = "Return a JSON object describing a fictional city with a name and a population.";

export async function runDecodePolicyBenchmark(
  options: DecodePolicyBenchmarkOptions = {},
): Promise<DecodePolicyBenchmarkArtifact> {
  const { iterations = 5, maxNewTokens = 64, ...loadOptions } = options;
  const cacheCapacity = loadOptions.cacheCapacity ?? 4096;
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new Error("Decode-policy benchmark iterations must be a positive integer");
  }
  const workloads: DecodePolicyWorkloadResult[] = [];
  const session = await loadGemmaGenerationSession({ ...loadOptions, cacheCapacity });
  try {
    await session.generate(PROMPT, { maxNewTokens: 8, reusePromptCache: false });
    workloads.push(await measureWorkload(session, "greedy-warm", iterations, PROMPT, {
      maxNewTokens,
      temperature: 0,
    }));
    workloads.push(await measureWorkload(session, "sampled-warm", iterations, PROMPT, {
      maxNewTokens,
      temperature: 0.8,
      topK: 40,
      topP: 0.95,
      repetitionPenalty: 1.05,
      repetitionWindow: 64,
      seed: 42,
    }));
    workloads.push(await measureWorkload(session, "json-schema-warm", iterations, JSON_PROMPT, {
      maxNewTokens,
      constraint: {
        type: "json-schema",
        schema: {
          type: "object",
          properties: { name: { type: "string" }, population: { type: "integer" } },
          required: ["name", "population"],
          additionalProperties: false,
        },
      },
    }));
  } finally {
    session.destroy();
  }

  const freshSamples: MeasuredSample[] = [];
  for (let iteration = 0; iteration < Math.min(iterations, 3); iteration += 1) {
    const fresh = await loadGemmaGenerationSession({ ...loadOptions, cacheCapacity });
    try {
      freshSamples.push(await measureOnce(fresh, PROMPT, { maxNewTokens, temperature: 0 }));
    } finally {
      fresh.destroy();
    }
  }
  workloads.push(summarize("greedy-fresh-session", freshSamples));

  return {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    environment: {
      userAgent: navigator.userAgent,
      adapterInfo: await adapterInfo(),
    },
    configuration: { cacheCapacity, iterations, maxNewTokens },
    workloads,
  };
}

interface MeasuredSample {
  timeToFirstTokenMs: number;
  interTokenLatencyMs: readonly number[];
  generatedTokens: number;
}

async function measureWorkload(
  session: GemmaGenerationSession,
  id: DecodePolicyWorkloadId,
  iterations: number,
  prompt: string,
  options: GemmaMeasuredGenerationOptions,
): Promise<DecodePolicyWorkloadResult> {
  const samples: MeasuredSample[] = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    samples.push(await measureOnce(session, prompt, options));
  }
  return summarize(id, samples);
}

async function measureOnce(
  session: GemmaGenerationSession,
  prompt: string,
  options: GemmaMeasuredGenerationOptions,
): Promise<MeasuredSample> {
  const { result, timing } = await session.generateMeasured(prompt, {
    ...options,
    reusePromptCache: false,
  });
  return {
    timeToFirstTokenMs: timing.timeToFirstTokenMs,
    interTokenLatencyMs: timing.interTokenLatencyMs,
    generatedTokens: result.generatedTokenIds.length,
  };
}

function summarize(
  id: DecodePolicyWorkloadId,
  samples: readonly MeasuredSample[],
): DecodePolicyWorkloadResult {
  return {
    id,
    iterations: samples.length,
    generatedTokens: samples.map((sample) => sample.generatedTokens),
    timeToFirstToken: distribution(samples.map((sample) => sample.timeToFirstTokenMs)),
    interTokenLatency: distribution(samples.flatMap((sample) => sample.interTokenLatencyMs)),
  };
}

function distribution(values: readonly number[]): DecodePolicyDistribution {
  const sorted = values.toSorted((left, right) => left - right);
  const at = (fraction: number): number => {
    if (sorted.length === 0) return Number.NaN;
    const position = (sorted.length - 1) * fraction;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const value = sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    return Math.round(value * 1000) / 1000;
  };
  return { medianMs: at(0.5), p95Ms: at(0.95), samples: sorted.length };
}

async function adapterInfo(): Promise<Record<string, string>> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  const info = adapter?.info;
  if (!info) return {};
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
  };
}
