import type { BenchmarkSummary, RawBenchmarkRun, SummaryRow } from "./types";

export interface BenchmarkReportArtifacts {
  rawResultsJsonl: string;
  summaryJson: string;
  reportMarkdown: string;
  reportCsv: string;
  reportHtml: string;
}

export function renderBenchmarkArtifacts(
  runs: readonly RawBenchmarkRun[],
  summary: BenchmarkSummary,
): BenchmarkReportArtifacts {
  return {
    rawResultsJsonl: runs.map((run) => JSON.stringify(run)).join("\n") + (runs.length ? "\n" : ""),
    summaryJson: `${JSON.stringify(summary, null, 2)}\n`,
    reportMarkdown: renderMarkdown(runs, summary),
    reportCsv: renderCsv(summary.rows),
    reportHtml: renderHtml(runs, summary),
  };
}

export function renderMarkdown(
  runs: readonly RawBenchmarkRun[],
  summary: BenchmarkSummary,
): string {
  const valid = runs.filter((run) => run.equalWorkEligible && !run.correctness.invalidOutput).length;
  const correctnessRows = correctnessSummary(runs);
  return `# Gemma 4 E2B Browser Benchmark\n\n` +
    `Generated: ${summary.generatedAt}\n\n` +
    `Browser mode: **${summary.environment.browserMode}** (never aggregated with the other mode)\n\n` +
    `Valid performance runs: ${valid}/${runs.length}. Correctness records include excluded runs.\n\n` +
    `## Method\n\n` +
    `Headline timings use common external boundaries: immediately before generation, first non-empty visible chunk, and generation completion. Runtime-native counters are retained only in raw results. Chunk intervals are not labeled token latency. Warm steady-state requires five warmups and 30 measured runs in the full profile.\n\n` +
    `## Performance\n\n` +
    `| Track | Mode | Cache | Runtime | Workload | n | Excluded | TTFT p50 ms | Total p50 ms | Decode p50 tok/s | Chars p50/s | 95% median CI (total ms) |\n` +
    `|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n` +
    summary.rows.map(markdownRow).join("\n") +
    `\n\n## Correctness\n\n` +
    `| Runtime | Workload | Runs | Invalid | Early stop | Prefix matched | Repetition flagged |\n` +
    `|---|---|---:|---:|---:|---:|---:|\n` +
    correctnessRows.map((row) =>
      `| ${row.runtimeId} | ${row.workloadId} | ${row.runs} | ${row.invalid} | ${row.early} | ${row.prefix} | ${row.repeated} |`
    ).join("\n") +
    `\n\n## Environment\n\n` +
    `- Browser: ${summary.environment.browserName} ${summary.environment.browserVersion}\n` +
    `- GPU: ${summary.environment.gpuAdapter}\n` +
    `- OS/device: ${summary.environment.operatingSystem}; ${summary.environment.physicalDevice}\n` +
    `- Power: ${summary.environment.powerSource}\n` +
    `- Git commit: ${summary.environment.gitCommit}\n` +
    `- Seed: ${summary.environment.benchmarkSeed}\n\n` +
    `## Limitations\n\n` +
    summary.limitations.map((limitation) => `- ${limitation}`).join("\n") + `\n`;
}

export function renderCsv(rows: readonly SummaryRow[]): string {
  const header = [
    "track", "browser_mode", "mode", "conversation_variant", "runtime_id", "workload_id", "valid_runs",
    "excluded_runs", "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms",
    "total_p50_ms", "total_p90_ms", "total_p95_ms", "decode_p50_tokens_per_second",
    "characters_p50_per_second", "total_median_ci95_low_ms", "total_median_ci95_high_ms",
  ];
  const body = rows.map((row) => [
    row.track, row.browserMode, row.mode, row.conversationVariant, row.runtimeId, row.workloadId, row.validRuns,
    row.excludedRuns, row.ttftMs?.p50, row.ttftMs?.p90, row.ttftMs?.p95,
    row.totalMs?.p50, row.totalMs?.p90, row.totalMs?.p95,
    row.aggregateDecodeTokensPerSecond?.p50, row.charactersPerSecond?.p50,
    row.totalMs?.medianConfidenceInterval95.low, row.totalMs?.medianConfidenceInterval95.high,
  ].map(csvValue).join(","));
  return [header.join(","), ...body].join("\n") + "\n";
}

export function renderHtml(runs: readonly RawBenchmarkRun[], summary: BenchmarkSummary): string {
  const chartRows = summary.rows.filter((row) => row.totalMs !== null).map((row) => ({
    label: `${row.runtimeId} ${row.workloadId} ${row.conversationVariant}`,
    ttft: row.ttftMs?.median ?? null,
    total: row.totalMs?.median ?? null,
    throughput: row.aggregateDecodeTokensPerSecond?.median ?? null,
    ci: row.totalMs?.medianConfidenceInterval95 ?? null,
    variation: row.totalMs?.coefficientOfVariation ?? null,
  }));
  const startupRows = runs.flatMap((run) => run.startup?.stages.flatMap((stage) =>
    stage.observable && stage.durationMs !== null
      ? [{ label: `${run.runtime.id} ${run.mode} ${stage.name}`, ready: stage.durationMs }]
      : []
  ) ?? []);
  const payload = escapeScript(JSON.stringify({ chartRows, startupRows }));
  return `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma Browser Benchmark</title><style>${htmlStyles()}</style></head><body><header><p>REPRODUCIBLE WEBGPU SUITE</p><h1>Gemma 4 E2B benchmark</h1><dl><div><dt>Mode</dt><dd>${escapeHtml(summary.environment.browserMode)}</dd></div><div><dt>Runs</dt><dd>${runs.length}</dd></div><div><dt>Seed</dt><dd>${summary.environment.benchmarkSeed}</dd></div><div><dt>GPU</dt><dd>${escapeHtml(summary.environment.gpuAdapter)}</dd></div></dl></header><main><section><h2>Time to first visible output</h2><div id="ttft" class="chart"></div></section><section><h2>Total latency with 95% median CI</h2><div id="latency" class="chart"></div></section><section><h2>Aggregate decode throughput</h2><div id="throughput" class="chart"></div></section><section><h2>Cold startup stages</h2><div id="startup" class="chart"></div></section><section><h2>Total latency coefficient of variation</h2><div id="variation" class="chart"></div></section><section><h2>Correctness</h2><p>${runs.filter((run) => !run.correctness.invalidOutput).length} valid outputs; ${runs.filter((run) => run.correctness.earlyTerminated).length} early terminations. Performance exclusions remain visible in raw results.</p></section></main><script>const data=${payload};${chartScript()}</script></body></html>\n`;
}

function markdownRow(row: SummaryRow): string {
  const ci = row.totalMs
    ? `${number(row.totalMs.medianConfidenceInterval95.low)}-${number(row.totalMs.medianConfidenceInterval95.high)}`
    : "n/a";
  return `| ${row.track} | ${row.mode} | ${row.conversationVariant} | ${row.runtimeId} | ${row.workloadId} | ${row.validRuns} | ${row.excludedRuns} | ${number(row.ttftMs?.p50)} | ${number(row.totalMs?.p50)} | ${number(row.aggregateDecodeTokensPerSecond?.p50)} | ${number(row.charactersPerSecond?.p50)} | ${ci} |`;
}

function correctnessSummary(runs: readonly RawBenchmarkRun[]) {
  const groups = new Map<string, RawBenchmarkRun[]>();
  for (const run of runs) {
    const key = `${run.runtime.id}\u0000${run.workload.id}`;
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  return [...groups.values()].map((group) => ({
    runtimeId: group[0].runtime.id,
    workloadId: group[0].workload.id,
    runs: group.length,
    invalid: group.filter((run) => run.correctness.invalidOutput).length,
    early: group.filter((run) => run.correctness.earlyTerminated).length,
    prefix: group.filter((run) => run.correctness.matchedExpectedPrefix).length,
    repeated: group.filter((run) => run.correctness.repeatedOutput).length,
  }));
}

function number(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : value.toFixed(2);
}

function csvValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function escapeHtml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function escapeScript(value: string): string {
  return value.replaceAll("<", "\\u003c");
}

function htmlStyles(): string {
  return `:root{font-family:"Avenir Next",Avenir,sans-serif;color:#17211d;background:#edf0e9}*{box-sizing:border-box}body{margin:0}header{padding:48px clamp(24px,6vw,96px);background:#13372e;color:#f6f3e8}header p{font:700 12px ui-monospace,monospace;letter-spacing:0}h1{font-family:Georgia,serif;font-size:clamp(38px,7vw,86px);font-weight:400;line-height:1;margin:20px 0 38px}dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:20px;margin:0}dt{font-size:12px;text-transform:uppercase}dd{margin:6px 0 0;font-weight:600}main{max-width:1200px;margin:auto;padding:36px 24px 80px}section{padding:28px 0;border-bottom:1px solid #aab4aa}h2{font-family:Georgia,serif;font-size:28px;font-weight:400}.chart{display:grid;gap:12px}.bar-row{display:grid;grid-template-columns:minmax(190px,1fr) 4fr 72px;gap:12px;align-items:center;font-size:13px}.bar-track{height:24px;background:#d8ddd5}.bar{height:100%;background:#d75238}.bar.ttft{background:#e5b83b}.bar.throughput{background:#247c6d}@media(max-width:640px){.bar-row{grid-template-columns:1fr 55px}.bar-track{grid-column:1/-1;grid-row:2}.bar-row span{overflow-wrap:anywhere}}`;
}

function chartScript(): string {
  return `function draw(id,rows,key,cls,showCi=false){const root=document.getElementById(id);const values=rows.map(r=>r[key]).filter(Number.isFinite);if(!values.length){root.textContent="No observations";return}const max=Math.max(...values);for(const row of rows){if(!Number.isFinite(row[key]))continue;const item=document.createElement("div");item.className="bar-row";const label=document.createElement("span");label.textContent=row.label;const track=document.createElement("div");track.className="bar-track";const bar=document.createElement("div");bar.className="bar "+cls;bar.style.width=(row[key]/max*100)+"%";track.append(bar);const value=document.createElement("strong");value.textContent=row[key].toFixed(1)+(showCi&&row.ci?" ["+row.ci.low.toFixed(1)+", "+row.ci.high.toFixed(1)+"]":"");item.append(label,track,value);root.append(item)}}draw("ttft",data.chartRows,"ttft","ttft");draw("latency",data.chartRows,"total","",true);draw("throughput",data.chartRows,"throughput","throughput");draw("startup",data.startupRows,"ready","ttft");draw("variation",data.chartRows,"variation","throughput");`;
}