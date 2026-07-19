# Reproducible Browser Benchmark Suite

The rigorous suite lives alongside the legacy compact proof. It does not change production
generation behavior or reinterpret the existing proof artifacts.

## Architecture

`src/benchmark/suite/types.ts` defines the shared `BenchmarkAdapter`, raw-run schema, environment,
schedule, and summary schema. Runtime adapters return generated text, runtime-specific token counts,
chunks, and separately named native metrics. The shared runner owns headline timestamps,
retokenization, correctness, equal-work exclusions, statistics, and reports.

The Playwright live orchestrator uses one persistent Chrome profile per runtime. Network-cold runs
clear the HTTP cache, Cache Storage, IndexedDB, local/session storage, service workers, and OPFS.
Cached-cold runs relaunch a fresh Chrome process and runtime against the retained profile. Warm and
conversation runs use loaded runtime pages and execute the recorded seeded randomized block order.

## Profiles

```sh
npm run benchmark:suite:smoke
npm run benchmark:suite:live-smoke
npm run benchmark:suite:full
npm run benchmark:suite:full:headed
```

The first command is a synthetic orchestration smoke check. Its numbers are not performance
evidence. The live smoke loads the real runtimes but reduces the workload and iteration count. Full
warm steady-state runs use five warmups and 30 measured iterations. Headed and headless results are
written to different directories and are rejected if mixed during aggregation.

Set `BENCHMARK_RUNTIMES=owned-webgpu,transformers-js` to select a subset, or `BENCHMARK_SEED` to
change the recorded order. The default seed is `20260717`.

Every completed run is appended immediately to `raw-results.jsonl`. If a browser or host failure
interrupts a full evaluation, resume the same deterministic schedule without repeating completed
runs:

```sh
BENCHMARK_SUITE_RESUME_DIR=benchmarks/suite/headless/<timestamp>-full \
  npm run benchmark:suite:resume
```

While a run is active, `progress.json` is updated after each run and phase transition. It reports
`completed`, `total`, `remaining`, `percentComplete`, `errors`, elapsed time, and a moving
`estimatedRemainingMs`, along with the current mode, runtime, and workload. Monitor it from another
terminal with:

```sh
tail -f benchmarks/suite/headless/<timestamp>-full/progress.json
```

The browser status page polls the same file once per second. Select a run with:

```text
http://127.0.0.1:5174/benchmark-status.html?run=benchmarks/suite/headless/<timestamp>-full
```

The top-level count uses `progressUnit: "measured-runs"`: it counts durable result rows, not model
loads, calibration, or warmups, so it is not a wall-clock percentage while setup is active. During
warmup, `phaseProgress` reports the current workload index, retry attempt, and completed warmup
generations. ETA remains null on resume until at least one new measured row establishes a rate.

The progress file is created by runs started with the current suite implementation. Older runs
still retain their durable `raw-results.jsonl` checkpoints and can be resumed normally.

If a runtime repeatedly fails warm setup after its valid cold-start rows have been checkpointed,
resume with `BENCHMARK_SKIP_WARM_RUNTIMES=<runtime-id>`. Its remaining warm and conversation rows
are reported as skipped with the reason retained in `progress.json` and report limitations; other
runtimes continue on the unchanged schedule. For example:

```sh
BENCHMARK_SKIP_WARM_RUNTIMES=transformers-js \
BENCHMARK_SUITE_RESUME_DIR=benchmarks/suite/headless/<timestamp>-full \
  npm run benchmark:suite:resume
```

Runtime generation errors are retained as invalid, equal-work-ineligible raw records. For warm and
conversation modes, the affected runtime is relaunched and warmed before the schedule continues.

## Workloads And Modes

The full deterministic matrix targets input/output token counts of `32/32`, `32/128`, `32/512`,
`153/128`, `256/128`, `639/128`, `1,024/128`, `4,096/128`, and `8,192/128` when the runtime context
supports the input plus output. Prompt padding is calibrated with each runtime's own input tokenizer
when exposed. Generated text is retokenized with that runtime's raw-output tokenizer path.

The four modes are:

- `network-cold-startup`: empty browser and origin storage, new process and runtime.
- `cached-cold-startup`: retained model cache, new Chrome process and runtime.
- `warm-steady-state`: loaded runtime after profile warmups.
- `conversation-cache`: separate fresh and prior-turn/KV-reused rows.

Cold startup uses three independent samples per runtime with the representative `32/32` workload.
Startup is a runtime/model property, so it is not multiplied by every generation workload. Warm and
conversation modes execute the complete matrix.

The request boundary is immediately before the runtime generation call. TTFT ends at the first
non-empty visible text chunk. Total latency ends when generation completes. Aggregate decode
throughput uses retokenized output after first visible output. Chunk intervals remain chunk
intervals unless a runtime guarantees one-token callbacks. Runtime-native counters are retained in
`raw-results.jsonl` and never substitute for external headline columns.

Runs that terminate materially short, produce empty/invalid output, or error remain in correctness
reporting but are excluded from equal-work performance aggregates. P95 is emitted only with at least
20 valid observations. Median 95% confidence intervals use a seeded bootstrap.

## Comparison Tracks

- Track A, `artifact-equivalent`, accepts pinned-source or demonstrated-equivalent artifacts.
- Track B, `best-available-stack`, compares complete deployable browser stacks and does not isolate
  library implementation performance.

Tokens per second remains tokenizer-specific when Track B artifacts differ. Characters per second,
bytes, exact text, character count, and runtime token counts are retained for interpretation.

## Outputs

Each live run writes a timestamped directory under `benchmarks/suite/<headed|headless>/`:

```text
raw-results.jsonl   one schema-versioned record per measured run
progress.json       live completion count, phase, errors, elapsed time, and ETA
summary.json        environment, exact schedule, limitations, and aggregate statistics
report.md           methodology, performance, correctness, environment, and limitations
report.csv          machine-readable headline aggregates
report.html         TTFT, total/CI, throughput, startup-stage, and variance charts
```

The latest synthetic structure check is under `benchmarks/suite-smoke/headless/`.

## Runtime API Limitations

- Owned WebGPU does not expose parsing, GPU upload, or shader compilation as separate loader stages.
  It also cannot suppress EOS in the public greedy path; short generations are excluded.
- Transformers.js loader stages below aggregate ready time are not separately observable. Streamer
  timestamps are JavaScript-visible events, not GPU completion timestamps.
- LiteRT-LM callbacks may contain multiple tokens and its public preview API does not expose
  arbitrary prompt tokenization or EOS suppression. Its nominal 4,096-token deterministic prompt
  expands to 9,711 LiteRT tokens and exceeds the 8,192-token context, so LiteRT 4K/8K rows are
  reported as unsupported. Native prefill/decode counters remain native.
- The pinned Hugging Face adapter loads immutable upstream source revision
  `158f16ae0f672943ca304d59c47c8e3a264e399e` through a Blob module and routes model requests to the
  exact local snapshot. The upstream API exposes aggregate readiness but not separate parse,
  upload, graph-creation, or compile stages, and fixes its default KV capacity at 8,192 tokens.
- Browser Battery Status may be unavailable. In that case power source is recorded as `unknown`.
- CDP can observe aggregate transfer activity during load, but runtime-internal parse/upload/compile
  boundaries remain null unless the runtime exposes them.

## Latest Full Evidence

The completed Chromium 150 / Apple M4 run is
[`2026-07-18T13-53-16-717Z-full`](suite/headless/2026-07-18T13-53-16-717Z-full/report.md): 2,094
measured records, 810 planned Transformers.js skips, 1,422 equal-work-eligible records, and 30
retained pinned-runtime errors. The report must be read with its correctness table: materially short
outputs remain durable evidence but are not promoted into equal-work performance aggregates.