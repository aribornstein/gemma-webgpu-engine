# Gemma WebGPU Engine

## Mission

Build an inspectable browser-native inference engine for `google/gemma-4-E2B-it-qat-mobile-transformers`. The engine owns its WebGPU execution pipeline, tokenizer/template integration, KV cache, sampling, streaming, profiling, and numerical validation. Its initial consumer is the Buza adaptive Levantine Arabic game.

This project replaces the opaque minimized JavaScript runtime, not Gemma itself. Upstream model artifacts remain independently versioned and are never committed without their applicable license and provenance metadata.

The product boundary is a complete generation engine over owned WebGPU kernels, not a greedy-only throughput demonstration. Optimizations must remain below the decoding-policy layer so the same transformer path continues to support full logits, seeded sampling, penalties, tokenizer-aware constraints, streaming, cancellation, and diagnostics. Greedy-only queueing and argmax-only LM-head shortcuts are out of scope even when they improve a narrow benchmark.

## Success criteria

- Equivalent greedy token IDs and acceptably close logits on a golden prompt suite.
- At least 1.5x warm decode throughput over the existing bundled runtime.
- Lower time to first token through persistent prefix KV caching.
- Real greedy, temperature, top-k, top-p, min-p, typical-p, repetition, frequency, and presence controls.
- Tokenizer-aware regex, JSON, and JSON Schema constraints that cannot be silently ignored.
- Deterministic seeded sampling, token streaming, cancellation, and no per-token GPU allocation.
- Explicit revision, tokenizer template, generation config, quantization, and file hashes.
- Device-loss recovery and actionable operator timings.

## Backbone and artifact policy

- Model ID: `google/gemma-4-E2B-it-qat-mobile-transformers`
- Family: Gemma 4 E2B instruction-tuned, quantization-aware mobile export
- Source: Hugging Face Hub
- Local destination: `public/models/gemma-4-e2b/` or browser cache
- Required manifest: repository revision, file list, SHA-256 hashes, license reference, tokenizer chat template, EOS IDs, and generation defaults

The browser downloader checks `public/models/gemma-4-e2b/model.safetensors` first, then range-fetches the pinned Hugging Face revision into resumable IndexedDB storage when the local file is absent. Authenticated Hugging Face access remains a local-tooling concern; tokens must never be committed or entered into the browser UI. The minimized Buza runtime is not copied.

## Architecture

1. **Artifacts:** manifest-driven range fetch, integrity verification, and IndexedDB cache.
2. **Tokenizer:** exact upstream tokenizer and chat-template rendering with golden vectors.
3. **Tensor store:** typed shapes, quantization metadata, immutable weights, reusable scratch arena.
4. **WebGPU graph:** explicit prefill and decode plans with pipeline caching and no hidden CPU fallback.
5. **KV cache:** preallocated/paged storage, grouped-query layout, prefix reuse, bounded eviction.
6. **Decoder:** logits processors, filters, seeded sampler, tokenizer-aware grammar constraints, EOS/stop sequences, streaming, cancellation.
7. **Telemetry:** load/prefill/decode timings, tokens per second, allocation counters, validation deltas, device limits.
8. **Reference harness:** compare tensors and logits with trusted Python/Transformers output.

## Kernel roadmap

| Milestone | Kernels and behavior | Gate |
|---|---|---|
| M0 | device, buffers, reductions, softmax, CPU sampling | error < 1e-5; deterministic tests |
| M1 | fp16/fp32 matmul, RMSNorm, residual, gating, RoPE | per-op reference parity |
| M2 | QAT dequantization fused with linear projections | >= 1.5x representative linear layer |
| M3 | grouped-query attention and contiguous KV cache | exact short-context token parity |
| M4 | full transformer prefill and decode graph | golden prompt parity |
| M5 | prefix cache, streaming, cancellation, regex/JSON/JSON Schema constraints | TTFT, valid-payload, and integration gates |
| M6 | speculative decoding and device-specific autotuning | measured improvement, never assumed |

## Decoding contract

`DecodingConfig` is stable public API. `temperature = 0` means greedy. Filters run in this order: repetition/frequency/presence penalties, temperature, top-k, min-p, typical-p, top-p, seeded multinomial sampling. Every generation records the effective configuration. Unsupported controls fail validation instead of being ignored.

`GenerationConstraint` is a separate optional contract. Constraints compile to minimized automata over exact tokenizer UTF-8 bytes, including partial code points across token boundaries. The current correctness path masks illegal entries in the persistent CPU logit readback before probability filters; moving sparse candidate masking to GPU is a measured performance milestone. EOS and configured stop tokens are available only in an accepting state. Regex support rejects assertions and backreferences; nested structured payloads use bounded JSON grammar or a closed JSON Schema subset. Every constrained result receives independent final validation, including AJV for schemas. Constraints may reduce generated tokens and retries, but are not assumed to reduce transformer execution time.

## Correctness and benchmark discipline

- Validate each kernel before composing it using random, adversarial, and real tensors.
- Record error bounds per dtype/operator; compare layer outputs, logits, greedy IDs, and seeded sequences.
- Use the same Chrome, adapter, power state, prompt, output length, and model artifacts for comparisons.
- Separate cold load, compilation, prefill, warm decode, and readback; report median, p95, tokens/second, TTFT, and memory estimate.
- Retain raw JSON artifacts. Performance changes cannot silently weaken correctness gates.

Current M2 correctness evidence uses the exact F32 layer-0 hidden state, RMS weight, SRQ activation and `sumA`, real Q/K/V weights, and logical Q/K/V outputs. The operator source is Hugging Face's `webml-community/gemma-4-webgpu-kernels` commit `158f16ae0f672943ca304d59c47c8e3a264e399e`; Buza merely vendors and invokes the byte-identical bundle with SHA-256 `0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62`. Hugging Face runs `DecodeRmsSrq` followed by `DecodeQkvProj`; the layer-0 logical output widths are Q `2048`, K `256`, and V `256`. The owned 256-thread RMS/SRQ kernel matches all 1,536 activation values and `sumA` exactly. Its output and sum buffers feed the owned fused QKV kernel directly, and the composed path matches every captured Q, K, and V value with maximum absolute and relative error `0`.

The owned K-only `DecodeQkNormRope` kernel uses the pinned 128-thread shared-memory reduction, full multiplier norm weight, and split-half RoPE, and matches all 256 captured outputs exactly on WebGPU. Owned QKV binds its K output subrange directly into this kernel, which writes to persistent K-cache position 10 with no CPU readback or GPU copy between operators. Raw Q/K/V and normalized K all retain maximum absolute and relative error `0`.

The owned fixed-32-lane-subgroup `Gemma4DecodeAttentionPartial` kernel now matches all 2,048 captured outputs bit-for-bit. Its single dispatch preserves the pinned runtime boundary: Q RMSNorm and split-half RoPE, grouped-query attention over the logical K/V cache prefix, flash partial accumulation, same-dispatch last-arriver merge, and output SRQ. The implementation uses ten persistent buffers totaling 313,408 bytes and allocates nothing per dispatch.

The owned fixed-32-lane `DecodeOprojNorm` kernel matches the pinned fused layer-0 boundary exactly. It performs the packed 4-bit `2048 -> 1536` output projection, projection SRQ, post-attention residual/RMS normalization, pre-FFN RMS normalization and F16 SRQ, and F32 input sum in one 192-workgroup dispatch. Updated hidden and sum match bit-for-bit, as do all 1,536 F16 output bits. The strict source capture was corrected to read the tensor actually bound as `sum2T` (`g4d-ff-suma`); the previous `g4d-n-suma` label referred to a different scalar buffer.

The reusable contiguous K/V cache owns fixed-capacity token-major buffers, validates contiguous writes and device limits, tracks logical length, supports metadata reset and GPU clearing, and exposes exact token/head offsets to producer kernels. The composed layer-0 path starts at the real hidden state and submits owned `DecodeRmsSrq`, QKV projection, K RMSNorm/RoPE, weightless V RMSNorm, fused attention, and `DecodeOprojNorm` as six ordered passes. QKV writes raw V directly into the current cache row for in-place normalization, K normalization writes directly into the matching key row, and attention output binds directly into O projection. This preserves the pinned runtime's previously implicit V RMSNorm step while removing both `StridedCopy` cache operations. Q, raw K, cached normalized K/V, attention output, updated hidden, F16 MLP input, and its sum are exact. The path uses 28 persistent buffers totaling 3,931,296 bytes, with no inter-kernel copies, readbacks, or dispatch allocations.

The remaining layer-0 MLP/PLE family is also owned: `DecodeGateUpNormPresrq`, `DecodeDownNormAddFused`, `DecodePleGateCodes`, and `DecodePleProjNormCodes`. The accelerated path dispatches all four kernels in one compute pass through shared persistent buffers. Final hidden and next-layer sum match bit-for-bit; every nonzero gate/up code, PLE gate value, and next-layer input value is exact. Chrome 149 differs from the Electron/Chrome 148 capture only in signed-zero bits (`+0` versus `-0`) at quantized boundaries, which is retained as an explicit diagnostic rather than counted as a numerical mismatch. The composed block owns 29 buffers totaling 15,091,800 bytes, performs no inter-kernel copies or readbacks, and allocates nothing per dispatch.

The complete layer-0 decode plan now joins the six attention-side dispatches and four MLP/PLE dispatches without staging copies. `DecodeOprojNorm` writes directly into the gate/up F16 input and sum buffers, while its updated hidden buffer remains the residual target for down projection and PLE projection. The ten-dispatch validation harness uses 55 GPU buffers totaling 19,026,168 bytes, including final readback resources, with zero copies, readbacks, or allocations between kernels. Final hidden and next-layer sum are bit-exact; layer-1 input has zero nonzero mismatches and 38 signed-zero differences across the Chrome 148/149 boundary.

The all-layer model planner is now validated directly against the immutable 375,400-byte safetensors header from revision `9fcec64df66cb1e4d972fc5cdc142afb25b2362c`. It requires 1,590 unique canonical tensors totaling 675,998,518 bytes across four exact execution profiles: 12 sliding/int4 layers, 3 full/int4 layers, 16 sliding/int2 layers, and 4 full/int2 layers. Full attention occurs at layers 4, 9, 14, 19, 24, 29, and 34 with 512-wide heads and 128 rotary dimensions; sliding attention uses 256-wide heads and a 512-token window. Layers 15–34 use a double-wide 12,288-element MLP with 2-bit packing, and only layers 0–14 carry K RMSNorm weights. Every dtype, shape, range, and byte length is checked before allocation.

The owned MLP/PLE block now derives its pipeline bit width from that materialized profile. The double-wide int2 gate/up kernel uses 12,288 rows, 96 packed U32 words per row, midpoint zero point 2, two rows per 32-lane subgroup, and 3,072 workgroups. The int2 down kernel retains 768 packed words and 384 workgroups while consuming four F16 activation vectors per word. A canonical-shape synthetic layer-15 test executes all four MLP/PLE dispatches in one submission with no WebGPU errors; a selected packed gate/up row produces exact F16 code 9, and a selected down row changes exactly one residual element when the PLE tail is neutralized. This is structural and numerical validation of the owned int2 profile, not a claim of exact layer-15 checkpoint parity.

Canonical layer weights can now be loaded by name in one readonly IndexedDB transaction per layer. Each payload is copied out of browser storage, length-checked against its planned descriptor, SHA-256 hashed, and retained with its exact source range. This bounds CPU staging to one layer rather than materializing all 676 MB at once and does not create, upgrade, or write the historical cache.

Canonical-to-kernel materialization is exact for the sliding/int4 profile. BF16 norms are promoted bit-preservingly to F32, F32 scales retain little-endian identity, packed U8 QAT storage is reinterpreted as U32 words, and signed I8 PLE weights are rebased by `+128` into the unsigned zero-point-128 codes consumed by WGSL. Q/K/V weights and row scales are concatenated in dispatch order; O-projection norms are `[post_attention, pre_feedforward]`; PLE norms are `[post_per_layer_input, next_input, layer_scalar]`. A reproducible verifier range-loads only the immutable header, layer 0, and layer 1 input norm, then compares 19 materialized resources against the exact captures. All comparisons are bit-exact across 18,576,466 layer-0 source bytes.

Both shared-buffer block constructors now accept materialized model weights directly. The existing ten-dispatch layer-0 GPU checkpoint uses that path for every model-owned weight and scale while retaining captured runtime activations, RoPE rows, cache prefix, PLE multiplier, and expected outputs. Its final hidden/input/sum guarantees and zero inter-kernel copy/readback/allocation properties remain unchanged.

The generic decode coordinator now has three execution modes. Layer 0 runs RMS/SRQ plus owned K/V for ten total layer dispatches; layers 1-14 consume the preceding PLE activation and run owned K/V in nine dispatches; layers 15-34 consume the preceding activation, project Q only, and read layer 13 or 14's external cache in seven dispatches. The planner-derived 35-layer stack therefore uses 276 dispatches instead of repeating the ten-dispatch layer-0 shape 35 times. Resource ownership distinguishes external activation/cache buffers from owned buffers, and readonly loading uses a two-layer CPU lookahead rather than retaining all 676 MB of materialized layer tensors.

The post-stack path is also owned. Layer 34's PLE projection applies `model.language_model.norm.weight` and the LM-head input scale directly to its final activation. A 2-bit, 262,144-row LM-head kernel then produces resident F32 logits in one dispatch, followed by a deterministic two-pass GPU argmax that chooses the lowest token ID on ties and reads back only eight bytes. Four input-preparation dispatches bring the complete greedy graph to 283 compute dispatches per evaluated token.

`GemmaGenerationSession` is the persistent public generation boundary. It loads the readonly checkpoint and tokenizer once and owns reusable GPU buffers and K/V caches. Automatic prefill uses sequential evaluation for prompts of at most 32 pending tokens and the exact fixed-32 graph repeatedly as `chunked-32` for longer prompts. `prefillStrategy: "fixed-32"`, `"chunked-32"`, and `"sequential"` remain available for explicit validation and equivalent measurements. Its implicit default remains exact greedy generation and keeps the deterministic eight-byte GPU argmax readback. Configured temperature, top-k, top-p, min-p, typical-p, repetition, frequency, and presence controls reuse one persistent full-logit readback buffer, then apply deterministic seeded CPU selection without per-token GPU allocation. Penalty history includes the prompt and accepted generated tokens, custom stop tokens are excluded from output, and every result records the effective config and stop reason. Accepted tokens are delivered through an awaited callback with immutable token snapshots and authoritative accumulated decoded text. `AbortSignal` is checked before and after prefill, decode, logits readback, and callback boundaries. Submitted GPU work is not interrupted mid-dispatch; cancellation rejects with the signal reason after that work finishes, and the next request clears every owner cache before reuse. The fixed graph preserves token-ID-0 padding, `M=32`, actual-last-row selection, and valid-length-only logical cache advancement. It reuses decode-owned weights and all 15 owner caches across 35 layers, including layer-13/14 shared K/V. Five golden cases match the pinned runtime in prompt IDs, every generated ID, EOS behavior, and final text, including Arabic and an exact 32-token prefill boundary. The vectors are retained in `src/runtime/gemma-golden.ts`.

The owned vision path is implemented from browser image decoding through language prefill. It performs deterministic resize and RGB patchification, pinned-range patch embedding, all 16 signed-I8 vision transformer layers with exact two-dimensional RoPE and bidirectional attention, 3x3 pooling, scale-free RMS normalization, and the F32 `768 -> 1536` language projection. Image soft tokens remain GPU-resident and replace PAD-derived language hidden rows before text prefill. Vision layers are loaded, executed, and destroyed sequentially to bound memory. Primitive, layer, full-tower, postprocess, tokenizer, browser preprocessing, abort-boundary, and maximum-2,520-patch tests pass. The answer-free dolphin example completed same-origin generation and accurately transcribed its visible caption and credit line without a CPU transformer fallback. The measured request spent 103.6 seconds in browser preprocessing plus vision encoding out of 110.9 seconds total, making vision the dominant optimization target. The console reports per-layer progress and separate preprocessing/encoding timing; live mid-tower cancellation, repeated-request resource stability, full-generation maximum geometry, and finer range-read/layer/projection timing remain open gates.

The model advertises 131,072 positions, while the production console now allocates and exposes the certified 32,768-position tier. Sliding-attention owners use fixed 512-position circular caches and full-attention owners use the configured logical capacity. Real 8K allocation, generation across the 512-position boundary, unaligned common-prefix reuse with hybrid chunked prefill, cancellation, post-cancellation recovery, exact-fit generation at position 8,192, and synchronous rejection at 8,193 pass on the cache-owning browser. The exact-fit request emitted one token with `chunked-32` in 89.1 seconds while retaining 898.5 MiB of GPU buffers.

The same-origin console now includes a durable long-context certification harness. It checkpoints load, retained memory, every 1,024 prefill rows, emitted tokens, exact-fit timing, overflow rejection, wrapped-prefix verification, cancellation, and terminal errors in versioned browser storage; completed artifacts survive reload and download as JSON. This prevents automation timeouts from erasing successful long runs.

32K certification passes on the stable production preview. The measured exact-fit request retained 2,849 buffers totaling 1,244,172,500 bytes, emitted token `1509` (`It`) with `chunked-32`, and completed in 670,105.2 ms, including 670,079.4 ms of prefill. The automatic follow-up safely rolled the wrapped sliding caches back one row, reused exactly 32,767 prompt tokens, reproduced the same output, and completed in 171.5 ms. A separate loaded session rejected 32,769 requested positions before prefill. Cancellation checkpointed cleanly at 31,744 rows, and the subsequent full run proves clean recovery. Wrapped sliding caches permit only the one-row rollback that remains physically valid; deeper rollback requiring overwritten rows is rejected. Text-only common-prefix reuse is implemented; multimodal requests reset caches because image identity is not yet part of the reuse contract.

128K certification also passes. The exact-fit run retained 2,849 buffers totaling 2,452,132,052 bytes, emitted token `1509` (`It`) with `chunked-32`, and completed in 4,424,216.2 ms, including 4,424,070.7 ms of prefill. Wrapped-prefix verification reused exactly 131,071 prompt tokens, reproduced the same output, and completed in 1,070.9 ms. The loaded session rejected 131,073 requested positions before prefill. A cancelled run checkpointed after 1,024 completed rows, and the clean subsequent exact-fit run proves recovery. This validates the model boundary while keeping the normal console at 32K because 128K retains about 2.28 GiB and requires roughly 74 minutes for this exact full-attention workload on the tested device.

The fixed-32 graph is composed from independently GPU-validated exact primitives: batched input preparation, staged-SRQ low-bit projection, cascade RMSNorm, split-half RoPE, mutable strided copy, tiled causal/windowed GQA attention, residual/scalar/LUT elementwise arithmetic, and code-backed dense PLE GEMV. The PLE kernels reconstruct the pinned `f32(i8) * row_scale` weights from the existing unsigned code buffers at load time, avoiding the pinned runtime's duplicate dense-float model representation. Non-final fixed blocks now submit asynchronously with a four-block fence, while 16/32-byte primitive uniforms use aligned shared arenas with standalone private-buffer fallback. This reduces the retained fixed graph from 2,678 to 1,600 GPU buffers and its additional buffer count over sequential from 1,209 to 131. Exact 32-token and 153-token outputs remain unchanged, and cancellation immediately after the first fixed block recovers cleanly on the same session. The latest alternating sweeps improve median TTFT by 5.5% at 32 tokens and 1.9% at 153 tokens, but regress p95 by 38.2% and 0.8%, respectively. The earlier 3.84-second chunked outlier is absent, but neither latency gate currently passes; [benchmarks/full-generation-prefill-batched-arena.electron-148.json](benchmarks/full-generation-prefill-batched-arena.electron-148.json) retains the raw evidence and makes no speedup claim.

Fixed-prefill stage profiling is now available through `generateMeasured(..., { profilePrefillStages: true })`. It uses GPU timestamp queries to report input, per-layer attention, feed-forward, PLE, and output boundaries without changing the normal one-pass production encoder. The 32/153/639-token target-selection sweep attributed 80.3-83.3% of fixed-prefill GPU time to feed-forward work. The resulting exact gate/up fusion replaces two staged SRQs and two projection dispatches with one staged SRQ and one dual-output projection per layer, removing 70 dispatches per fixed block. Independent int4/int2 GPU tests match both standalone outputs bit-for-bit; retained 32- and 153-token model goldens also match exactly. Five alternating samples per mode improve median/p95 profiled prefill by 21.8%/32.4% at 32 tokens, 15.9%/13.6% at 153, and 17.0%/28.7% at 639. The default fused graph retains 1,599 buffers / 942,411,076 bytes versus 1,600 / 942,476,612 for the load-time `separate` diagnostic fallback. [benchmarks/full-generation-prefill-gate-up-fused.electron-148.json](benchmarks/full-generation-prefill-gate-up-fused.electron-148.json) retains the raw samples and comparison metadata.

The remaining language-runtime plan is complete. A ten-sample unprofiled gate/up confirmation improves median/p95 TTFT by 10.7%/23.9% at 32 tokens, 10.9%/9.7% at 153, and 13.1%/15.6% at 639. Exact fused RMS residual and residual-scale epilogues are also promoted, removing 140 dispatches per fixed block while retaining raw-bit parity. Their unprofiled median/p95 improvements are 6.1%/3.7%, 3.6%/2.8%, and 3.1%/2.3% at the same boundaries. An eight-block submission window, shared-SRQ QKV, and direct strided PLE input all preserve exact output but fail at least one median/p95 promotion gate, so their prior production paths remain the defaults. The final graph passes six retained/canonical text cases, first-block cancellation and clean same-session recovery, 40 focused language tests, typecheck, and production build. [benchmarks/full-generation-language-optimization-finish.electron-148.json](benchmarks/full-generation-language-optimization-finish.electron-148.json) retains the raw evidence and decisions.

Two model-wide defects were found through this parity work. First, nonzero layers were incorrectly bound to layer 0's captured GELU lookup tables. The pinned table builder is scale-dependent and uses a carefully rounded F32 tanh approximation; generating gate and PLE tables from every materialized layer's own output scales makes the first divergent layer-1 gate/up boundary bit-exact. Second, the generic attention shader allocated 256-element workgroup Q-normalization and output arrays even when full-attention profiles use 512-wide heads. That out-of-bounds access made layer 4 and later output nondeterministic. Compiling those arrays at the profile head dimension makes repeated BOS evaluations produce identical logits and bit-identical owner K/V rows across all 15 cache-owning layers.

Performance evidence keeps host and GPU timing separate and compares only equivalent work. The source-equivalent Q-only `QatMatMul scalar_presrq` path measures `0.023593 ms` GPU-timestamp median, versus `0.603 ms` for the original owned Q kernel, a `25.6x` Q-only improvement. For the equivalent fused boundary, the owned combined-storage `DecodeQkvProj presrq` kernel measures `0.030802 ms` median while Hugging Face's pinned operator measures `0.01769472 ms` in the same warm Chrome tab. Both use 100 repeated dispatches per timestamp sample. The owned fused path is `1.74x` slower, so no speedup is claimed and the M2 performance gate remains open.

Owned RMS/SRQ measures `0.015073 ms` median over 100 dispatches encoded in one compute pass. The shared-storage RMS-to-QKV sequence uses ten persistent buffers, no intermediate CPU readback, and zero per-dispatch allocation. In an equivalent same-tab comparison using 100 ordered two-pass pairs per timestamp sample, owned measures `0.034734 ms` median and the pinned Hugging Face sequence measures `0.02097152 ms`. The owned pair is `1.66x` slower, so no speedup is claimed and the M2 performance gate remains open.

In headless Chrome 149, owned K norm/RoPE measures `0.003932 ms` median over 100 dispatches per timestamp sample. The direct QKV-to-K-cache sequence measures `0.037356 ms` per ordered pair, uses twelve persistent buffers, performs no intermediate readback or copy, and allocates nothing per dispatch. This remains owned-only Chrome 149 evidence.

The equivalent QKV/K-norm-RoPE pair was measured separately in the cache-owning Electron 42.5.0 / Chrome 148 tab. Three alternating rounds produced 30 samples per implementation with 100 ordered two-pass pairs per sample. Owned measures `0.020972 ms` median and `0.02687 ms` p95; pinned Hugging Face measures `0.02228224 ms` median and `0.02555904 ms` p95. Owned is `1.06x` faster at median but `1.05x` slower at p95. The environment is recorded explicitly, all raw samples are retained, and the `1.5x` project gate remains open.

In headless Chrome 149, owned fused attention measures `0.043202 ms` median and `0.056177 ms` p95 over 20 dispatches per timestamp sample. This remains owned-only evidence. In the cache-owning Electron 42.5.0 / Chrome 148 tab, three alternating equivalent rounds produced 30 samples per implementation: owned and pinned Hugging Face occupy the same median timer bucket (`0.00983 ms` and `0.0098304 ms`). Owned p95 is `0.00983 ms` versus `0.0131072 ms`, but no meaningful speedup is claimed at this timer resolution. Both artifacts retain all raw samples and keep their browser environments separate.

## Repository layout

```text
src/runtime/       decoding and future generation sessions
src/webgpu/        device ownership and WGSL operators
src/model/         manifests, tokenizer, config, and tensor loading
src/reference/     CPU/reference operators and comparisons
tests/             decoding, browser kernels, and golden vectors
tools/             artifact download, hashing, and reference export
```

## Execution plan

1. **Vision hardening and optimization:** validate live mid-tower cancellation, repeated-request resource stability, and full generation at 2,520 patches; split range-read, per-layer, pooling, and projection timings; then add an immutable multimodal tensor cache and optimize the measured 103.6-second vision path without weakening parity.
2. **Multi-turn hardening:** role-preserving text history and the canonical Gemma 4 template are exposed; extend edit and cache-invalidation gates, then define content-hash ownership before permitting multimodal prefix reuse.
3. **Audio:** implement pinned audio preprocessing, tower/bridge materialization, owned WebGPU execution, soft-token insertion, parity tests, cancellation, and progress reporting without changing the historical language cache.
4. **Video:** confirm the pinned processor contract, add deterministic frame sampling and timestamps, reuse the validated vision path where the checkpoint requires it, and enforce multi-frame context and memory bounds.
5. **Performance:** continue evidence-gated kernel work below the decoding-policy layer. The measured pass rejected exact QKV source-layout, retained the existing gate/up and down kernels because they already beat equivalent pinned-HF timings, and promoted a block-major full-logit LM head after all-logit error, exact greedy/seeded/constraint sequences, retained memory, median, and p95 passed. A row-cooperative O-projection alternate is exact and improves 200-dispatch aggregate median/p95 by 5.37%/8.33%, but remains non-default until full-generation canonical and end-to-end A/B gates pass. Keep all logits available for sampling and constraints; do not add greedy-only queueing or argmax-only handoff. The prior kernels remain load-time fallbacks.
6. **Reliability and release:** recover from WebGPU device loss, stress resource destruction and repeated requests, handle interrupted pinned-range reads, run the full target-browser matrix, and retain reproducible correctness and benchmark artifacts.

Speculative decoding and device-specific autotuning remain later measured milestones. No optimization advances automatically unless it preserves the corresponding token, tensor, constraint, memory, and cancellation gates.

## Non-goals for the first proof

- Arbitrary transformer support or browser training.
- Replacing deterministic game authority with model output.
- Claiming speedups without same-device measurements.
- Fine-tuning before runtime and integration variables are controlled.
