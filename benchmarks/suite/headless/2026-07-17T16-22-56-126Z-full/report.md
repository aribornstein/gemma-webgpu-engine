# Gemma 4 E2B Browser Benchmark

Generated: 2026-07-17T23:11:44.980Z

Browser mode: **headless** (never aggregated with the other mode)

Valid performance runs: 612/1008. Correctness records include excluded runs.

## Method

Headline timings use common external boundaries: immediately before generation, first non-empty visible chunk, and generation completion. Runtime-native counters are retained only in raw results. Chunk intervals are not labeled token latency. Warm steady-state requires five warmups and 30 measured runs in the full profile.

## Performance

| Track | Mode | Cache | Runtime | Workload | n | Excluded | TTFT p50 ms | Total p50 ms | Decode p50 tok/s | Chars p50/s | 95% median CI (total ms) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| artifact-equivalent | cached-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-256-output-128 | 30 | 0 | 3529.90 | 5256.20 | 74.16 | 139.65 | 5160.90-5391.55 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-128 | 30 | 0 | 473.70 | 2931.65 | 51.63 | 250.71 | 2851.50-3008.30 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-512 | 30 | 0 | 425.50 | 10337.50 | 51.54 | 166.97 | 9992.50-10622.90 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-256-output-128 | 30 | 0 | 19.15 | 2361.80 | 54.22 | 311.20 | 2213.20-2468.40 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-128 | 30 | 0 | 17.90 | 2212.55 | 57.88 | 332.20 | 2110.80-2344.60 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-512 | 30 | 0 | 14.60 | 9580.90 | 53.32 | 180.15 | 9424.00-9659.70 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | network-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-256-output-128 | 30 | 0 | 3124.40 | 5110.05 | 64.68 | 143.64 | 4999.00-5316.95 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-128 | 30 | 0 | 456.75 | 2971.90 | 51.16 | 247.32 | 2844.80-3075.90 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-512 | 30 | 0 | 479.70 | 10335.15 | 51.37 | 167.00 | 9911.90-11066.25 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | cached-cold-startup | not-applicable | litert-lm-web | input-32-output-32 | 3 | 0 | 144.10 | 733.50 | 52.81 | 249.49 | 731.10-758.60 |
| best-available-stack | cached-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | cached-cold-startup | not-applicable | transformers-js | input-32-output-32 | 3 | 0 | 253.70 | 1249.60 | 31.39 | 146.45 | 1229.80-1733.70 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-1024-output-128 | 30 | 0 | 2825.75 | 6518.90 | 35.45 | 112.75 | 6382.15-6673.50 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-256-output-128 | 30 | 0 | 798.45 | 4226.70 | 37.55 | 173.90 | 4167.98-4300.55 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-256-output-128 | 30 | 0 | 3529.90 | 5256.20 | 74.16 | 139.65 | 5138.05-5391.55 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-128 | 30 | 0 | 179.00 | 3663.35 | 36.31 | 200.64 | 3631.41-3684.95 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-128 | 30 | 0 | 473.70 | 2931.65 | 51.63 | 250.71 | 2851.50-3008.30 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-32 | 30 | 0 | 141.05 | 923.40 | 39.06 | 198.18 | 908.20-978.50 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-512 | 30 | 0 | 145.30 | 14575.05 | 35.35 | 201.92 | 14435.65-14703.40 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-512 | 30 | 0 | 425.50 | 10337.50 | 51.54 | 166.97 | 9975.45-10622.90 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-256-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-256-output-128 | 30 | 0 | 19.15 | 2361.80 | 54.22 | 311.20 | 2213.20-2468.55 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-128 | 30 | 0 | 17.90 | 2212.55 | 57.88 | 332.20 | 2104.20-2345.70 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-512 | 30 | 0 | 154.35 | 14476.80 | 35.68 | 203.29 | 14347.05-14620.50 |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-512 | 30 | 0 | 14.60 | 9580.90 | 53.32 | 180.15 | 9424.00-9646.75 |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | network-cold-startup | not-applicable | litert-lm-web | input-32-output-32 | 3 | 0 | 141.90 | 731.40 | 52.48 | 250.21 | 725.20-754.40 |
| best-available-stack | network-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | network-cold-startup | not-applicable | transformers-js | input-32-output-32 | 3 | 0 | 642.90 | 1647.60 | 30.81 | 111.07 | 1320.80-2017.00 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-1024-output-128 | 30 | 0 | 2473.60 | 5430.65 | 43.47 | 135.34 | 5350.55-5599.35 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-256-output-128 | 30 | 0 | 762.30 | 4170.25 | 37.85 | 176.25 | 4100.60-4230.15 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-256-output-128 | 30 | 0 | 3124.40 | 5110.05 | 64.68 | 143.64 | 5000.20-5316.95 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-128 | 30 | 0 | 160.85 | 3826.35 | 34.64 | 192.09 | 3723.05-3880.50 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-128 | 30 | 0 | 456.75 | 2971.90 | 51.16 | 247.32 | 2844.80-3073.75 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-32 | 30 | 0 | 132.85 | 945.35 | 38.26 | 193.59 | 906.40-981.30 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-512 | 30 | 0 | 155.65 | 14195.30 | 36.31 | 207.32 | 13783.50-14979.85 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-512 | 30 | 0 | 479.70 | 10335.15 | 51.37 | 167.00 | 9952.00-11082.80 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |

## Correctness

| Runtime | Workload | Runs | Invalid | Early stop | Prefix matched | Repetition flagged |
|---|---|---:|---:|---:|---:|---:|
| owned-webgpu | input-32-output-32 | 96 | 0 | 96 | 30 | 0 |
| transformers-js | input-32-output-32 | 6 | 0 | 0 | 6 | 0 |
| litert-lm-web | input-32-output-32 | 96 | 0 | 0 | 96 | 0 |
| litert-lm-web | input-32-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-32-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-32-output-512 | 90 | 0 | 90 | 0 | 0 |
| litert-lm-web | input-32-output-512 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-256-output-128 | 90 | 0 | 0 | 30 | 0 |
| litert-lm-web | input-256-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-1024-output-128 | 90 | 0 | 90 | 0 | 0 |
| litert-lm-web | input-1024-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-4096-output-128 | 90 | 0 | 90 | 0 | 0 |

## Environment

- Browser: Chromium 150
- GPU: apple / metal-3
- OS/device: Darwin 25.5.0 arm64; Aris-MacBook-Air.local
- Power: battery
- Git commit: 15b82bb92909d5e4a57124a37ef2f85ecd412320
- Seed: 20260717

## Limitations

- Track B compares complete deployable browser stacks and does not isolate library implementation performance.
- Token counts are runtime-tokenizer-specific; cross-artifact tokens/second is not a unit-invariant measure.
- LiteRT-LM exposes chunk callbacks rather than guaranteed token callbacks, so only chunk intervals are reported.
- Pinned Hugging Face current-browser execution is unavailable because its pinned browser bundle is absent.
- The 8,192-token input row is executed only by runtimes whose configured context can also hold the requested output.
- transformers-js warm runtime disabled by BENCHMARK_SKIP_WARM_RUNTIMES after external failure
- Runtime prompt calibration unavailable for input-32-output-32: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:48:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:31:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:24:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-32-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:48:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:31:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:24:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-32-output-512: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:48:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:31:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:24:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-256-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:48:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:31:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:24:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-1024-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:48:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:31:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:24:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
