import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { renderE2BPerformanceProofMarkdown } from
  "../src/benchmark/e2b-performance-proof.ts";

const artifactPath = path.join(
  process.cwd(),
  "benchmarks/e2b-performance-proof.chrome.json",
);
const reportPath = path.join(process.cwd(), "benchmarks/BENCHMARK_RESULTS.md");
const captured = JSON.parse(await readFile(artifactPath, "utf8"));

if (!captured?.artifact || captured.artifact.schemaVersion !== 1) {
  throw new Error(`Unsupported E2B performance artifact: ${artifactPath}`);
}

const firstOwned = captured.raw?.owned?.[0];
const ownedRuntime = captured.artifact.runtimes.find(
  (runtime) => runtime.id === "owned-webgpu",
);
if (firstOwned && ownedRuntime && !ownedRuntime.notes.some((note) =>
  note.startsWith("Fresh session load")
)) {
  ownedRuntime.notes.splice(
    1,
    0,
    `Fresh session load from the local pinned file: ${firstOwned.load.sessionLoadMs} ms.`,
  );
}

for (const testCase of captured.artifact.cases) {
  const rawCase = captured.raw?.liteRtLm?.cases?.find(
    (candidate) => candidate.id === testCase.id,
  );
  const result = testCase.results.find(
    (candidate) => candidate.runtimeId === "litert-lm-web",
  );
  if (!rawCase || !result) continue;
  result.metrics.itlMs = distribution(
    rawCase.samples.flatMap((sample) => sample.externallyObserved.chunkIntervalMs),
  );
  const nativeTtftMs = distribution(
    rawCase.samples.map((sample) => sample.native.timeToFirstTokenMs),
  )?.median;
  result.note = `ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is ${nativeTtftMs} ms.`;
}

await writeFile(artifactPath, `${JSON.stringify(captured, null, 2)}\n`);
await writeFile(
  reportPath,
  renderE2BPerformanceProofMarkdown(captured.artifact),
);

console.log(`Wrote ${path.relative(process.cwd(), reportPath)}`);

function distribution(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return {
    median: round(percentile(sorted, 0.5)),
    p95: round(percentile(sorted, 0.95)),
  };
}

function percentile(sorted, quantile) {
  return sorted[
    Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * quantile) - 1))
  ];
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}