export type E2BRuntimeId =
  | "owned-webgpu"
  | "pinned-hugging-face-webgpu"
  | "transformers-js"
  | "litert-lm-web";

export type E2BEvidenceStatus =
  | "same-device-measured"
  | "prior-same-device-measured"
  | "published-reference"
  | "blocked";

export type E2BArtifactEquivalence =
  | "pinned-source-equivalent"
  | "model-family-only"
  | "unverified";

export interface E2BRuntimeEvidence {
  id: E2BRuntimeId;
  label: string;
  version: string;
  modelArtifact: string;
  artifactEquivalence: E2BArtifactEquivalence;
  status: E2BEvidenceStatus;
  notes: readonly string[];
}

export interface E2BMetricDistribution {
  median: number;
  p95: number;
}

export interface E2BCaseMetrics {
  ttftMs: E2BMetricDistribution | null;
  itlMs: E2BMetricDistribution | null;
  tpotMs: E2BMetricDistribution | null;
  decodeTokensPerSecond: number | null;
  totalMs: E2BMetricDistribution | null;
}

export interface E2BCaseRuntimeResult {
  runtimeId: E2BRuntimeId;
  evidenceStatus: E2BEvidenceStatus;
  exactOutputMatch: boolean | null;
  generatedTokens: number | null;
  metrics: E2BCaseMetrics;
  note?: string;
}

export interface E2BPerformanceCase {
  id: string;
  promptTokens: number;
  expectedOutputTokens: number | null;
  results: readonly E2BCaseRuntimeResult[];
}

export interface E2BPublishedReference {
  runtimeId: E2BRuntimeId;
  device: string;
  promptTokens: number;
  outputTokens: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
  timeToFirstTokenSeconds: number;
  modelSizeMb: number;
  memoryMb: number;
  source: string;
}

export interface E2BPerformanceProofArtifact {
  schemaVersion: 1;
  capturedAt: string;
  status: "partial" | "complete";
  model: {
    id: string;
    revision: string;
  };
  environment: {
    userAgent: string;
    adapter: string;
    device: string;
  };
  methodology: readonly string[];
  runtimes: readonly E2BRuntimeEvidence[];
  cases: readonly E2BPerformanceCase[];
  publishedReferences: readonly E2BPublishedReference[];
}

const REQUIRED_RUNTIME_IDS: readonly E2BRuntimeId[] = [
  "owned-webgpu",
  "pinned-hugging-face-webgpu",
  "transformers-js",
  "litert-lm-web",
];

export function canClaimE2BBroadSuperiority(
  artifact: E2BPerformanceProofArtifact,
): boolean {
  if (artifact.status !== "complete" || artifact.cases.length === 0) return false;
  if (!REQUIRED_RUNTIME_IDS.every((runtimeId) => {
    const runtime = artifact.runtimes.find((candidate) => candidate.id === runtimeId);
    return runtime?.status === "same-device-measured" &&
      runtime.artifactEquivalence === "pinned-source-equivalent";
  })) return false;

  return artifact.cases.every((testCase) => {
    const owned = testCase.results.find((result) => result.runtimeId === "owned-webgpu");
    if (!owned || owned.exactOutputMatch !== true) return false;
    return REQUIRED_RUNTIME_IDS.filter((runtimeId) => runtimeId !== "owned-webgpu")
      .every((runtimeId) => {
        const competitor = testCase.results.find((result) => result.runtimeId === runtimeId);
        return competitor?.exactOutputMatch === true &&
          winsAllRequiredMetrics(owned.metrics, competitor.metrics);
      });
  });
}

export function renderE2BPerformanceProofMarkdown(
  artifact: E2BPerformanceProofArtifact,
): string {
  const broadClaim = canClaimE2BBroadSuperiority(artifact);
  const lines = [
    "# Gemma 4 E2B Browser Performance Proof",
    "",
    `Captured: ${artifact.capturedAt}`,
    "",
    `Status: **${artifact.status}**`,
    "",
    "## Verdict",
    "",
    broadClaim
      ? "The owned WebGPU runtime passed every required same-device median and p95 gate against all equivalent competitor artifacts."
      : "No blanket performance-superiority claim is supported. The tables below report the measured wins, losses, evidence gaps, and artifact-equivalence limits directly.",
    "",
    "## Environment",
    "",
    `- Device: ${artifact.environment.device}`,
    `- GPU adapter: ${artifact.environment.adapter}`,
    `- Browser: ${artifact.environment.userAgent}`,
    `- Model: \`${artifact.model.id}\` at \`${artifact.model.revision}\``,
    "",
    "## Runtime Identity",
    "",
    "| Runtime | Version | Evidence | Artifact equivalence | Model artifact |",
    "| --- | --- | --- | --- | --- |",
    ...artifact.runtimes.map((runtime) =>
      `| ${runtime.label} | ${runtime.version} | ${runtime.status} | ${runtime.artifactEquivalence} | ${runtime.modelArtifact} |`
    ),
    "",
    ...artifact.runtimes.flatMap((runtime) => [
      `### ${runtime.label}`,
      "",
      ...runtime.notes.map((note) => `- ${note}`),
      "",
    ]),
    "## Same-Device Results",
    "",
  ];

  for (const testCase of artifact.cases) {
    lines.push(
      `### ${testCase.id}`,
      "",
      `Prompt tokens: ${testCase.promptTokens}; expected output tokens: ${testCase.expectedOutputTokens ?? "not comparable"}.`,
      "",
      "| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |",
      "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
      ...testCase.results.map((result) => {
        const metrics = result.metrics;
        return `| ${runtimeLabel(artifact, result.runtimeId)} | ${formatDistribution(metrics.ttftMs)} | ${formatDistribution(metrics.itlMs)} | ${formatDistribution(metrics.tpotMs)} | ${formatNumber(metrics.decodeTokensPerSecond)} | ${formatDistribution(metrics.totalMs)} | ${formatMatch(result.exactOutputMatch)} |`;
      }),
      "",
    );
    const owned = testCase.results.find((result) => result.runtimeId === "owned-webgpu");
    if (owned) {
      lines.push(
        "Owned relative delta; positive means the owned runtime is faster, negative means it is slower.",
        "",
        "| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        ...testCase.results.filter((result) => result.runtimeId !== "owned-webgpu")
          .map((competitor) =>
            `| ${runtimeLabel(artifact, competitor.runtimeId)}${competitor.exactOutputMatch === false ? " (output differs)" : ""} | ${formatLatencyDelta(owned.metrics.ttftMs, competitor.metrics.ttftMs)} | ${formatLatencyDelta(owned.metrics.itlMs, competitor.metrics.itlMs)} | ${formatLatencyDelta(owned.metrics.tpotMs, competitor.metrics.tpotMs)} | ${formatThroughputDelta(owned.metrics.decodeTokensPerSecond, competitor.metrics.decodeTokensPerSecond)} | ${formatLatencyDelta(owned.metrics.totalMs, competitor.metrics.totalMs)} |`
          ),
        "",
      );
    }
    const notes = testCase.results.flatMap((result) =>
      result.note ? [`- ${runtimeLabel(artifact, result.runtimeId)}: ${result.note}`] : []
    );
    if (notes.length > 0) lines.push(...notes, "");
  }

  lines.push("## Published Reference", "");
  if (artifact.publishedReferences.length === 0) {
    lines.push("No external reference rows were included.", "");
  } else {
    lines.push(
      "Published rows are context only and are never used for same-device speedup claims.",
      "",
      "| Runtime | Device | Prefill | Decode | TTFT | Model size | Memory | Workload |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
      ...artifact.publishedReferences.map((reference) =>
        `| ${runtimeLabel(artifact, reference.runtimeId)} | ${reference.device} | ${reference.prefillTokensPerSecond} tok/s | ${reference.decodeTokensPerSecond} tok/s | ${reference.timeToFirstTokenSeconds} s | ${reference.modelSizeMb} MB | ~${reference.memoryMb} MB | ${reference.promptTokens} prefill / ${reference.outputTokens} decode |`
      ),
      "",
      ...artifact.publishedReferences.map((reference) => `Source: ${reference.source}`),
      "",
    );
  }

  lines.push("## Methodology", "", ...artifact.methodology.map((item) => `- ${item}`), "");
  return `${lines.join("\n")}\n`;
}

function winsAllRequiredMetrics(
  owned: E2BCaseMetrics,
  competitor: E2BCaseMetrics,
): boolean {
  const latencyPairs = [
    [owned.ttftMs, competitor.ttftMs],
    [owned.itlMs, competitor.itlMs],
    [owned.tpotMs, competitor.tpotMs],
    [owned.totalMs, competitor.totalMs],
  ] as const;
  if (!latencyPairs.every(([ownedValue, competitorValue]) =>
    ownedValue !== null && competitorValue !== null &&
    ownedValue.median < competitorValue.median && ownedValue.p95 < competitorValue.p95
  )) return false;
  return owned.decodeTokensPerSecond !== null &&
    competitor.decodeTokensPerSecond !== null &&
    owned.decodeTokensPerSecond > competitor.decodeTokensPerSecond;
}

function runtimeLabel(
  artifact: E2BPerformanceProofArtifact,
  runtimeId: E2BRuntimeId,
): string {
  return artifact.runtimes.find((runtime) => runtime.id === runtimeId)?.label ?? runtimeId;
}

function formatDistribution(value: E2BMetricDistribution | null): string {
  return value === null ? "N/A" : `${formatNumber(value.median)} / ${formatNumber(value.p95)} ms`;
}

function formatNumber(value: number | null): string {
  return value === null ? "N/A" : new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 3,
  }).format(value);
}

function formatMatch(value: boolean | null): string {
  return value === null ? "Not comparable" : value ? "Yes" : "No";
}

function formatLatencyDelta(
  owned: E2BMetricDistribution | null,
  competitor: E2BMetricDistribution | null,
): string {
  return owned === null || competitor === null
    ? "N/A"
    : formatPercent((competitor.median - owned.median) / competitor.median * 100);
}

function formatThroughputDelta(owned: number | null, competitor: number | null): string {
  return owned === null || competitor === null
    ? "N/A"
    : formatPercent((owned - competitor) / competitor * 100);
}

function formatPercent(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}