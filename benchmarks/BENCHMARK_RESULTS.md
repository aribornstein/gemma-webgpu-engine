# Gemma 4 E2B Browser Performance Proof

Captured: 2026-07-17T11:46:54.890Z

Status: **partial**

## Verdict

No blanket performance-superiority claim is supported. The tables below report the measured wins, losses, evidence gaps, and artifact-equivalence limits directly.

## Environment

- Device: MacBook Air (local benchmark host)
- GPU adapter: apple / metal-3
- Browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/150.0.0.0 Safari/537.36
- Model: `google/gemma-4-E2B-it-qat-mobile-transformers` at `9fcec64df66cb1e4d972fc5cdc142afb25b2362c`

## Runtime Identity

| Runtime | Version | Evidence | Artifact equivalence | Model artifact |
| --- | --- | --- | --- | --- |
| Owned WebGPU | workspace | same-device-measured | pinned-source-equivalent | https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors (2458111846 bytes) |
| Pinned Hugging Face WebGPU | 158f16ae0f672943ca304d59c47c8e3a264e399e | prior-same-device-measured | pinned-source-equivalent | https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors |
| Transformers.js | 4.2.0 | same-device-measured | model-family-only | onnx-community/gemma-4-E2B-it-ONNX@9f4bef82ea6e296bc69f8a2f5939f73af81b07a6 (q4f16) |
| LiteRT-LM Web | 0.14.0 | same-device-measured | model-family-only | https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm |

### Owned WebGPU

- Loads the pinned mobile-QAT safetensors artifact through repository-owned kernels.
- Fresh session load from the local pinned file: 7068.9 ms.
- Retained GPU buffers for the first case: 848039236 bytes across 1599 buffers; driver and pipeline allocations are excluded.

### Pinned Hugging Face WebGPU

- Reuses the existing exact-output same-device artifact; it was not rerun because its vendored upstream bundle is not present in this workspace.
- Previous clean cached load: 4716.6 ms.

### Transformers.js

- Fresh processor and model load: 309681.9 ms.
- The ONNX Community q4f16 export is derived from the Gemma 4 E2B instruction model, but it is not the file-identical mixed 2/4/8-bit mobile-QAT safetensors artifact used by the owned runtime.
- Gemma4ForCausalLM intentionally selects only embed_tokens and decoder_model_merged; vision and audio encoder sessions are excluded from load time and generation.
- Token timestamps are captured by TextStreamer after each generated token reaches JavaScript. They include runtime dispatch, readback, sampling, and callback overhead.

### LiteRT-LM Web

- Fresh engine load: 199088.5 ms.
- The official Web .litertlm file is a specially optimized text-only export, not the pinned mobile-QAT safetensors file.
- Externally observed callback intervals are chunk intervals. They are not promoted to per-token ITL because the preview API does not guarantee one token per callback.
- Native LiteRT-LM TTFT and decode throughput are retained as supplemental runtime counters; cross-runtime claims use externally equivalent boundaries where available.

## Same-Device Results

### short-greeting

Prompt tokens: 16; expected output tokens: 2.

| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Owned WebGPU | 211.2 / 248.9 ms | 14.4 / 16.7 ms | 14.4 / 16.7 ms | 66.89 | 239.7 / 282.1 ms | Yes |
| Pinned Hugging Face WebGPU | 202 / 202.45 ms | 13.8 / 16.86 ms | 14.867 / 14.867 ms | 67.265 | 269.8 / 281.23 ms | Yes |
| Transformers.js | 82.3 / 82.8 ms | 33.4 / 37.1 ms | 33.4 / 37.1 ms | 30.769 | 146.7 / 153.5 ms | Yes |
| LiteRT-LM Web | 53.1 / 53.9 ms | 19.3 / 19.6 ms | 13.333 / 13.667 ms | 75 | 91.8 / 93.1 ms | Yes |

Owned relative delta; positive means the owned runtime is faster, negative means it is slower.

| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Hugging Face WebGPU | -4.6% | -4.3% | +3.1% | -0.6% | +11.2% |
| Transformers.js | -156.6% | +56.9% | +56.9% | +117.4% | -63.4% |
| LiteRT-LM Web | -297.7% | +25.4% | -8% | -10.8% | -161.1% |

- Pinned Hugging Face WebGPU: Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.
- Transformers.js: Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.
- LiteRT-LM Web: ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is 64.333 ms.

### arithmetic

Prompt tokens: 21; expected output tokens: 2.

| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Owned WebGPU | 379 / 397.5 ms | 22.7 / 24.1 ms | 22.7 / 24.1 ms | 45.524 | 421.2 / 442.1 ms | Yes |
| Pinned Hugging Face WebGPU | 209.9 / 211.79 ms | 13.5 / 15.12 ms | 13.8 / 13.8 ms | 72.464 | 277 / 279.79 ms | Yes |
| Transformers.js | 114.6 / 115.4 ms | 32.2 / 33 ms | 32.2 / 33 ms | 31.447 | 177.9 / 179 ms | Yes |
| LiteRT-LM Web | 63.2 / 63.6 ms | 20.5 / 20.8 ms | 14.333 / 14.667 ms | 69.767 | 103.8 / 105.7 ms | Yes |

Owned relative delta; positive means the owned runtime is faster, negative means it is slower.

| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Hugging Face WebGPU | -80.6% | -68.1% | -64.5% | -37.2% | -52.1% |
| Transformers.js | -230.7% | +29.5% | +29.5% | +44.8% | -136.8% |
| LiteRT-LM Web | -499.7% | -10.7% | -58.4% | -34.7% | -305.8% |

- Pinned Hugging Face WebGPU: Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.
- Transformers.js: Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.
- LiteRT-LM Web: ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is 74.333 ms.

### arabic

Prompt tokens: 18; expected output tokens: 5.

| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Owned WebGPU | 324.8 / 327.3 ms | 18.8 / 22.2 ms | 19.475 / 22.025 ms | 50.59 | 421.6 / 437.8 ms | Yes |
| Pinned Hugging Face WebGPU | 214.9 / 218.05 ms | 10.95 / 14.09 ms | 11.55 / 11.55 ms | 86.58 | 313.3 / 328.42 ms | Yes |
| Transformers.js | 110.1 / 112.8 ms | 31.2 / 32.1 ms | 31.35 / 31.55 ms | 31.928 | 265.6 / 270.2 ms | Yes |
| LiteRT-LM Web | 60.6 / 60.6 ms | 19.8 / 20.6 ms | 16.667 / 17.333 ms | 60 | 158.2 / 162.3 ms | Yes |

Owned relative delta; positive means the owned runtime is faster, negative means it is slower.

| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Hugging Face WebGPU | -51.1% | -71.7% | -68.6% | -41.6% | -34.6% |
| Transformers.js | -195% | +39.7% | +37.9% | +58.5% | -58.7% |
| LiteRT-LM Web | -436% | +5.1% | -16.8% | -15.7% | -166.5% |

- Pinned Hugging Face WebGPU: Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.
- Transformers.js: Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.
- LiteRT-LM Web: ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is 74.667 ms.

### longer-instruction

Prompt tokens: 19; expected output tokens: 11.

| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Owned WebGPU | 316.6 / 348.6 ms | 17.8 / 19.3 ms | 17.65 / 18.63 ms | 55.885 | 509.4 / 542.8 ms | Yes |
| Pinned Hugging Face WebGPU | 219.7 / 221.86 ms | 10.55 / 13.415 ms | 11.227 / 11.227 ms | 89.074 | 386.4 / 392.7 ms | Yes |
| Transformers.js | 109.2 / 110.6 ms | 31.2 / 31.7 ms | 31.191 / 31.318 ms | 32.051 | 484.7 / 485.2 ms | No |
| LiteRT-LM Web | 63.4 / 64.5 ms | 20 / 22 ms | 18.583 / 19.25 ms | 53.812 | 284.5 / 292.9 ms | No |

Owned relative delta; positive means the owned runtime is faster, negative means it is slower.

| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Hugging Face WebGPU | -44.1% | -68.7% | -57.2% | -37.3% | -31.8% |
| Transformers.js (output differs) | -189.9% | +42.9% | +43.4% | +74.4% | -5.1% |
| LiteRT-LM Web (output differs) | -399.4% | +11% | +5% | +3.9% | -79.1% |

- Pinned Hugging Face WebGPU: Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.
- Transformers.js: Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.
- LiteRT-LM Web: ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is 78.583 ms.

### prefill-32-boundary

Prompt tokens: 32; expected output tokens: 1.

| Runtime | TTFT median / p95 | ITL median / p95 | TPOT median / p95 | Decode tok/s | Total median / p95 | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Owned WebGPU | 518.1 / 574 ms | N/A | N/A | 50.083 | 539.7 / 596.3 ms | Yes |
| Pinned Hugging Face WebGPU | 226.9 / 238.33 ms | N/A | N/A | N/A | 287.9 / 303.02 ms | Yes |
| Transformers.js | 113.6 / 113.8 ms | N/A | N/A | 31.447 | 145.5 / 145.6 ms | Yes |
| LiteRT-LM Web | 73.3 / 74.5 ms | N/A | 11.5 / 12.5 ms | 86.956 | 94.5 / 96.8 ms | Yes |

Owned relative delta; positive means the owned runtime is faster, negative means it is slower.

| Competitor | TTFT median | ITL median | TPOT median | Decode tok/s | Total median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned Hugging Face WebGPU | -128.3% | N/A | N/A | N/A | -87.5% |
| Transformers.js | -356.1% | N/A | N/A | +59.3% | -270.9% |
| LiteRT-LM Web | -606.8% | N/A | N/A | -42.4% | -471.1% |

- Pinned Hugging Face WebGPU: Prior same-device/browser-version evidence; TPOT is derived from aggregate steady throughput and is not a measured p95 distribution.
- Transformers.js: Token timing uses TextStreamer callbacks. Output equality is checked against the owned mobile-QAT golden text, but differences do not invalidate the separately labeled model-family performance row.
- LiteRT-LM Web: ITL is an externally observed callback-interval proxy because LiteRT-LM does not guarantee one token per callback. TPOT and decode tok/s use native benchmark counters; native TTFT median is 81.5 ms.

## Published Reference

Published rows are context only and are never used for same-device speedup claims.

| Runtime | Device | Prefill | Decode | TTFT | Model size | Memory | Workload |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LiteRT-LM Web | MacBook Pro M4 Max / WebGPU | 4853 tok/s | 73 tok/s | 1.09 s | 2008 MB | ~1800 MB | 1024 prefill / 256 decode |

Source: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm

## Methodology

- Owned WebGPU, Transformers.js, and LiteRT-LM ran sequentially in fresh browser contexts; every runtime-owned session, conversation, engine, and context was destroyed before the next runtime.
- Each live case used one warmup followed by three measured greedy generations with a fresh prompt cache or conversation.
- Externally observed TTFT and total wall time are primary. Runtime-native counters are supplemental unless their timing boundaries are demonstrably equivalent.
- The pinned Hugging Face row is prior same-device evidence from Chrome 148/Electron 42 and is not treated as a current-browser measurement.
- Transformers.js uses the pinned ONNX Community q4f16 text-only export. It is a model-family comparison, not file-identical model execution.
- LiteRT-LM uses Google's specially optimized text-only Web .litertlm artifact. It is a model-family comparison, not file-identical model execution.
- No browser runtime exposes a portable retained GPU-memory measurement with equivalent boundaries; memory is therefore omitted from same-device speedup gates.

