# Gemma 4 E2B Browser Benchmark

Generated: 2026-07-19T01:27:52.857Z

Browser mode: **headless** (never aggregated with the other mode)

Valid performance runs: 1422/2094. Correctness records include excluded runs.

## Method

Headline timings use common external boundaries: immediately before generation, first non-empty visible chunk, and generation completion. Runtime-native counters are retained only in raw results. Chunk intervals are not labeled token latency. Warm steady-state requires five warmups and 30 measured runs in the full profile.

## Performance

| Track | Mode | Cache | Runtime | Workload | n | Excluded | TTFT p50 ms | Total p50 ms | Decode p50 tok/s | Chars p50/s | 95% median CI (total ms) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| artifact-equivalent | cached-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | cached-cold-startup | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 7097.90 | 8825.65 | 76.67 | 83.28 | 8551.25-9043.25 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-153-output-128 | 30 | 0 | 2274.50 | 4145.70 | 69.17 | 177.05 | 4040.00-4324.15 |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1744.35 | 3655.10 | 69.10 | 201.10 | 3398.30-3822.20 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-256-output-128 | 30 | 0 | 3538.30 | 5320.25 | 72.79 | 137.97 | 5231.70-5746.60 |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2521.35 | 4243.45 | 74.12 | 173.21 | 4172.45-4370.80 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-128 | 30 | 0 | 448.85 | 2180.65 | 73.70 | 337.06 | 2150.75-2200.60 |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 328.80 | 2049.25 | 74.16 | 358.67 | 2033.10-2118.15 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-32-output-512 | 30 | 0 | 484.80 | 8928.45 | 60.71 | 193.31 | 8694.30-9296.60 |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 406.50 | 8618.10 | 63.28 | 200.51 | 8476.40-9040.25 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-4096-output-128 | 30 | 0 | 33136.05 | 36658.60 | 35.90 | 20.05 | 35977.28-36971.40 |
| artifact-equivalent | conversation-cache | fresh | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | fresh | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 5154.30 | 7002.25 | 70.88 | 104.97 | 6711.75-7589.20 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 7099.80 | 9164.50 | 63.84 | 80.20 | 8873.95-9397.20 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-153-output-128 | 30 | 0 | 17.40 | 2411.05 | 53.10 | 304.87 | 2294.40-2551.30 |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1430.25 | 3249.20 | 71.20 | 226.23 | 3085.45-3457.70 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-256-output-128 | 30 | 0 | 17.75 | 2332.50 | 54.89 | 315.11 | 2196.75-2609.75 |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2377.05 | 4067.20 | 74.43 | 180.72 | 4005.95-4166.05 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-128 | 30 | 0 | 16.75 | 1730.00 | 74.12 | 424.86 | 1713.35-1753.55 |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 279.75 | 2025.50 | 73.03 | 362.88 | 1991.70-2052.25 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-32-output-512 | 30 | 0 | 14.25 | 8823.15 | 57.90 | 195.63 | 8348.25-9091.75 |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 308.15 | 8451.85 | 63.29 | 348.21 | 8257.80-8680.90 |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | conversation-cache | reused | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 5086.60 | 7229.80 | 63.41 | 101.77 | 6816.35-7673.25 |
| artifact-equivalent | network-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | network-cold-startup | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 8009.00 | 9928.15 | 65.98 | 74.03 | 9878.00-10021.40 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-153-output-128 | 30 | 0 | 2247.30 | 4184.40 | 66.84 | 175.41 | 4123.40-4309.50 |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1922.40 | 3981.85 | 66.68 | 184.59 | 3774.10-4108.50 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-256-output-128 | 30 | 0 | 2992.00 | 4846.50 | 70.01 | 151.45 | 4724.60-4948.02 |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2236.25 | 3964.40 | 72.78 | 185.40 | 3942.05-4060.55 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-128 | 30 | 0 | 401.70 | 2546.85 | 59.16 | 288.60 | 2500.20-2610.30 |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 308.30 | 2205.00 | 67.53 | 333.34 | 2082.05-2366.20 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-32-output-512 | 30 | 0 | 413.10 | 10124.90 | 52.56 | 170.49 | 9606.50-10406.85 |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 284.55 | 8106.40 | 65.27 | 213.16 | 7986.15-8246.75 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-4096-output-128 | 30 | 0 | 31650.65 | 34996.25 | 38.91 | 21.00 | 34549.20-35369.85 |
| artifact-equivalent | warm-steady-state | not-applicable | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| artifact-equivalent | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 4858.35 | 6664.75 | 70.69 | 110.28 | 6560.15-6858.90 |
| best-available-stack | cached-cold-startup | not-applicable | litert-lm-web | input-32-output-32 | 3 | 0 | 140.20 | 756.50 | 50.16 | 241.90 | 755.70-759.80 |
| best-available-stack | cached-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | cached-cold-startup | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | cached-cold-startup | not-applicable | transformers-js | input-32-output-32 | 3 | 0 | 782.60 | 1782.50 | 31.00 | 102.67 | 1693.30-1788.30 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-1024-output-128 | 30 | 0 | 2418.90 | 5051.25 | 48.42 | 145.51 | 4914.15-5198.70 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 7097.90 | 8825.65 | 76.67 | 83.28 | 8551.25-9043.25 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-153-output-128 | 30 | 0 | 555.35 | 4203.65 | 35.45 | 174.85 | 3911.40-4361.80 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-153-output-128 | 30 | 0 | 2274.50 | 4145.70 | 69.17 | 177.05 | 4044.60-4324.15 |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1744.35 | 3655.10 | 69.10 | 201.10 | 3398.30-3822.20 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-256-output-128 | 30 | 0 | 844.55 | 4187.65 | 37.68 | 175.52 | 4089.75-4230.48 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-256-output-128 | 30 | 0 | 3538.30 | 5320.25 | 72.79 | 137.97 | 5231.70-5746.60 |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2521.35 | 4243.45 | 74.12 | 173.21 | 4172.45-4376.10 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-128 | 30 | 0 | 149.45 | 3048.25 | 43.82 | 241.12 | 3024.10-3106.80 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-128 | 30 | 0 | 448.85 | 2180.65 | 73.70 | 337.06 | 2150.75-2200.30 |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 328.80 | 2049.25 | 74.16 | 358.67 | 2036.25-2118.15 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-32 | 30 | 0 | 137.05 | 812.65 | 46.42 | 225.31 | 778.65-838.00 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-32-output-512 | 30 | 0 | 151.30 | 13120.85 | 39.45 | 224.35 | 12767.30-13618.25 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-32-output-512 | 30 | 0 | 484.80 | 8928.45 | 60.71 | 193.31 | 8709.65-9296.60 |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 406.50 | 8618.10 | 63.28 | 200.51 | 8468.70-9054.25 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-4096-output-128 | 30 | 0 | 33136.05 | 36658.60 | 35.90 | 20.05 | 36007.70-36971.40 |
| best-available-stack | conversation-cache | fresh | litert-lm-web | input-639-output-128 | 30 | 0 | 2035.65 | 5395.05 | 39.22 | 136.24 | 4652.05-5605.98 |
| best-available-stack | conversation-cache | fresh | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | fresh | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 5154.30 | 7002.25 | 70.88 | 104.97 | 6711.75-7622.65 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 7099.80 | 9164.50 | 63.84 | 80.20 | 8873.95-9397.20 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-153-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-153-output-128 | 30 | 0 | 17.40 | 2411.05 | 53.10 | 304.87 | 2294.40-2532.75 |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1430.25 | 3249.20 | 71.20 | 226.23 | 3085.45-3480.55 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-256-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-256-output-128 | 30 | 0 | 17.75 | 2332.50 | 54.89 | 315.11 | 2196.75-2609.30 |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2377.05 | 4067.20 | 74.43 | 180.72 | 4003.45-4187.10 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-128 | 30 | 0 | 16.75 | 1730.00 | 74.12 | 424.86 | 1714.75-1753.20 |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 279.75 | 2025.50 | 73.03 | 362.88 | 1991.70-2052.40 |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-32-output-512 | 30 | 0 | 138.80 | 12978.15 | 39.80 | 226.77 | 12732.80-13756.65 |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-32-output-512 | 30 | 0 | 14.25 | 8823.15 | 57.90 | 195.63 | 8348.25-9082.55 |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 308.15 | 8451.85 | 63.29 | 348.21 | 8257.80-8680.90 |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | litert-lm-web | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | conversation-cache | reused | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 5086.60 | 7229.80 | 63.41 | 101.77 | 6816.35-7673.25 |
| best-available-stack | network-cold-startup | not-applicable | litert-lm-web | input-32-output-32 | 3 | 0 | 144.50 | 753.40 | 50.79 | 242.90 | 750.50-755.00 |
| best-available-stack | network-cold-startup | not-applicable | owned-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | network-cold-startup | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | network-cold-startup | not-applicable | transformers-js | input-32-output-32 | 3 | 0 | 537.90 | 1575.70 | 29.82 | 116.14 | 1482.30-1583.80 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-1024-output-128 | 30 | 0 | 2925.30 | 5932.50 | 41.85 | 123.89 | 5886.00-6146.05 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-1024-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-1024-output-128 | 30 | 0 | 8009.00 | 9928.15 | 65.98 | 74.03 | 9851.40-10018.15 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-153-output-128 | 30 | 0 | 570.00 | 4263.95 | 35.26 | 172.38 | 3997.50-4465.60 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-153-output-128 | 30 | 0 | 2247.30 | 4184.40 | 66.84 | 175.41 | 4123.40-4309.50 |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-153-output-128 | 30 | 0 | 1922.40 | 3981.85 | 66.68 | 184.59 | 3761.10-4100.20 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-256-output-128 | 30 | 0 | 731.80 | 4074.80 | 38.47 | 180.38 | 3880.65-4132.15 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-256-output-128 | 30 | 0 | 2992.00 | 4846.50 | 70.01 | 151.45 | 4724.60-4947.95 |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-256-output-128 | 30 | 0 | 2236.25 | 3964.40 | 72.78 | 185.40 | 3942.05-4065.90 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-128 | 30 | 0 | 129.50 | 3106.50 | 42.72 | 236.61 | 3042.25-3159.60 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-128 | 30 | 0 | 401.70 | 2546.85 | 59.16 | 288.60 | 2500.20-2611.81 |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-128 | 30 | 0 | 308.30 | 2205.00 | 67.53 | 333.34 | 2082.05-2366.20 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-32 | 30 | 0 | 110.75 | 785.80 | 45.96 | 232.88 | 777.85-793.70 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-32 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-32-output-512 | 30 | 0 | 123.30 | 12879.95 | 40.08 | 228.50 | 12444.20-13056.10 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-32-output-512 | 30 | 0 | 413.10 | 10124.90 | 52.56 | 170.49 | 9606.50-10397.20 |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-32-output-512 | 30 | 0 | 284.55 | 8106.40 | 65.27 | 213.16 | 7986.15-8246.75 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-4096-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-4096-output-128 | 30 | 0 | 31650.65 | 34996.25 | 38.91 | 21.00 | 34549.20-35369.85 |
| best-available-stack | warm-steady-state | not-applicable | litert-lm-web | input-639-output-128 | 30 | 0 | 1804.45 | 4764.95 | 42.62 | 154.25 | 4569.15-4829.85 |
| best-available-stack | warm-steady-state | not-applicable | owned-webgpu | input-639-output-128 | 0 | 30 | n/a | n/a | n/a | n/a | n/a |
| best-available-stack | warm-steady-state | not-applicable | pinned-hugging-face-webgpu | input-639-output-128 | 30 | 0 | 4858.35 | 6664.75 | 70.69 | 110.28 | 6555.20-6853.35 |

## Correctness

| Runtime | Workload | Runs | Invalid | Early stop | Prefix matched | Repetition flagged |
|---|---|---:|---:|---:|---:|---:|
| owned-webgpu | input-32-output-32 | 96 | 0 | 96 | 30 | 0 |
| transformers-js | input-32-output-32 | 6 | 0 | 0 | 6 | 0 |
| litert-lm-web | input-32-output-32 | 96 | 0 | 0 | 96 | 0 |
| pinned-hugging-face-webgpu | input-32-output-32 | 96 | 0 | 96 | 96 | 0 |
| pinned-hugging-face-webgpu | input-32-output-128 | 90 | 0 | 0 | 90 | 0 |
| litert-lm-web | input-32-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-32-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-32-output-512 | 90 | 0 | 90 | 0 | 0 |
| litert-lm-web | input-32-output-512 | 90 | 0 | 0 | 90 | 0 |
| pinned-hugging-face-webgpu | input-32-output-512 | 90 | 0 | 0 | 30 | 0 |
| pinned-hugging-face-webgpu | input-153-output-128 | 90 | 0 | 0 | 90 | 0 |
| litert-lm-web | input-153-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-153-output-128 | 90 | 0 | 0 | 30 | 0 |
| litert-lm-web | input-256-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-256-output-128 | 90 | 0 | 0 | 30 | 0 |
| pinned-hugging-face-webgpu | input-256-output-128 | 90 | 0 | 0 | 90 | 0 |
| litert-lm-web | input-639-output-128 | 90 | 0 | 0 | 90 | 0 |
| pinned-hugging-face-webgpu | input-639-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-639-output-128 | 90 | 0 | 90 | 0 | 0 |
| pinned-hugging-face-webgpu | input-1024-output-128 | 90 | 0 | 0 | 90 | 0 |
| litert-lm-web | input-1024-output-128 | 90 | 0 | 0 | 90 | 0 |
| owned-webgpu | input-1024-output-128 | 90 | 0 | 90 | 0 | 0 |
| owned-webgpu | input-4096-output-128 | 90 | 0 | 90 | 0 | 0 |
| pinned-hugging-face-webgpu | input-4096-output-128 | 90 | 30 | 30 | 60 | 0 |

## Environment

- Browser: Chromium 150
- GPU: apple / metal-3
- OS/device: Darwin 25.5.0 arm64; Aris-MacBook-Air.local
- Power: external-power
- Git commit: b161f3ebe17e7395d12363e84f045296af35f1b9
- Seed: 20260717

## Limitations

- Track B compares complete deployable browser stacks and does not isolate library implementation performance.
- Token counts are runtime-tokenizer-specific; cross-artifact tokens/second is not a unit-invariant measure.
- LiteRT-LM exposes chunk callbacks rather than guaranteed token callbacks, so only chunk intervals are reported.
- The pinned Hugging Face runtime is loaded from its immutable upstream browser bundle because the raw host serves it with a non-module MIME type.
- The 8,192-token input row is executed only by runtimes whose configured context can also hold the requested output.
- The public session loader does not separately expose parsing, GPU upload, or shader compilation durations.
- EOS suppression is not supported by the owned greedy path; early termination is recorded and excluded from equal-work throughput.
- TextStreamer callbacks are JavaScript-visible token/text events, not GPU completion timestamps.
- The loader does not expose parse, graph creation, GPU upload, and shader compilation as separate stages.
- Streaming callbacks may contain multiple tokens; only stream chunk intervals are reported externally.
- The public API does not expose an arbitrary model-tokenizer encode operation; native prefill/decode counts are retained separately.
- EOS suppression is not exposed by the preview API; early termination is recorded and excluded from equal-work throughput.
- Runtime prompt calibration unavailable for input-32-output-32: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- The runtime is loaded from the immutable upstream bundle at 158f16ae0f672943ca304d59c47c8e3a264e399e.
- The upstream API exposes aggregate readiness only, not separate parse, graph creation, GPU upload, or shader compilation stages.
- The upstream runtime fixes its default KV cache capacity at 8,192 tokens.
- transformers-js warm runtime unavailable: Error: transformers-js warm setup failed after 3 attempts
- Runtime prompt calibration unavailable for input-32-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-32-output-512: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-153-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-256-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-639-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
- Runtime prompt calibration unavailable for input-1024-output-128: Error: page.evaluate: Error: LiteRT-LM does not expose arbitrary input tokenizer encoding
    at LiteRtLmBenchmarkAdapter.countTokens (http://127.0.0.1:5174/src/benchmark/suite/adapters/litert-lm.ts:126:9)
    at findClosestPrompt (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:50:31)
    at calibrateWorkloadForRuntime (http://127.0.0.1:5174/src/benchmark/suite/workloads.ts:33:23)
    at Module.calibrateBenchmarkWorkload (http://127.0.0.1:5174/src/benchmark/suite/browser-harness.ts:20:9)
    at eval (eval at evaluate (:303:30), <anonymous>:4:22)
    at async <anonymous>:329:30
