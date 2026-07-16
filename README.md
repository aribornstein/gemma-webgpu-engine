# Gemma WebGPU Engine

An inspectable WebGPU inference runtime for the Gemma 4 E2B QAT mobile backbone, built for low-latency browser generation and complete decoding control.

See [PROJECT.md](PROJECT.md) for architecture, correctness gates, model provenance, and the kernel roadmap.

## Start

```bash
npm install
npm run dev
```

Chrome with WebGPU is required for kernel benchmarks. Sampling tests do not require a GPU.
The first screen is the generation console. Because IndexedDB is origin-scoped, serve it from the
same scheme, host, and port that owns `safetensors-cache-v1`; the console inventories the origin and
keeps model loading disabled when that cache is absent. The Buza host exposes the permanent
same-origin route `http://localhost:8753/gemma-engine.html`. When this browser has no cached model,
that route enables **Initialize cache** and uses Buza's existing client-side downloader before
loading the owned engine; no inference backend is involved.

```bash
npm run typecheck
npm run test:unit
npm run test:gpu
npm run build
```

## Generate

The owned runtime now exposes a persistent prompt-to-text session:

```ts
import { loadGemmaGenerationSession } from "./src/runtime/gemma-session";

const session = await loadGemmaGenerationSession({ cacheCapacity: 512 });
try {
	const controller = new AbortController();
	const result = await session.generate("Say hi in one short sentence.", {
		maxNewTokens: 16,
		temperature: 0.8,
		topK: 40,
		topP: 0.95,
		seed: 42,
		constraint: { type: "regex", pattern: "Hi!" },
		signal: controller.signal,
		onToken: ({ text }) => {
			console.log(text);
		},
	});
	console.log(result.text);
} finally {
	session.destroy();
}
```

The console owns one persistent session and exposes exact greedy or seeded sampling, every penalty
and probability control, custom stop IDs, regex/JSON/closed-schema constraints, streamed output,
and cancellation. Editable examples populate the prompt and matching controls for greedy, sampling,
regex, bounded JSON, and closed JSON Schema generation. Its telemetry reports session load, retained GPU-buffer memory, TTFT, median
decode latency, total latency, prefill route, and stop reason from the same measured runtime path.
`Decode tok/s` divides post-first-token decode evaluations by their measured decode time, while
`Overall tok/s` divides every emitted token by the complete request time including prefill.
The cache-origin browser gate loaded the model, generated the longer golden exactly, cancelled after
one streamed token while preserving partial output, reused the interrupted session safely, and then
generated and parsed `{"ok":true}` through the closed JSON Schema editor.

The session loads the checkpoint once, reuses all GPU resources, resets or truncates its owned K/V
caches between requests, applies the pinned chat template, and decodes until an EOS token,
configured stop token, or the requested limit. Automatic prefill uses sequential evaluation for at
most 32 pending prompt tokens and repeats the exact fixed-32 graph as `chunked-32` for longer
prompts. Explicit `fixed-32`, `chunked-32`, and `sequential` strategies remain available for
validation and equivalent benchmarking.
Temperature, top-k, top-p, min-p, typical-p, repetition, frequency, and presence controls are
available directly on `generate()`, with one seeded PRNG per request. The implicit default remains
exact greedy generation for backward compatibility. Neutral greedy requests retain the existing
eight-byte GPU argmax readback; sampling and penalty requests reuse one persistent full-logit GPU
readback buffer and perform deterministic CPU token selection without per-token GPU allocation.
Constrained requests use that same resident readback, mask it to tokenizer-byte-trie candidates,
and then run the configured deterministic selection pipeline. Unconstrained greedy routing is
unchanged.
Accepted tokens stream through an awaited `onToken` callback with the token ID, token index, a
snapshot of all generated IDs, and the authoritative accumulated decoded text. Awaiting the handler
provides backpressure. `AbortSignal` cancellation is cooperative at safe runtime boundaries:
already-submitted GPU work finishes, then generation rejects with `signal.reason` before another
token is emitted or evaluated. Interrupted cache state is never reused; the next request clears all
owner caches before prefill. Every completed result records its resolved decoding configuration and
stop reason. IndexedDB is origin-scoped, so this
module must run in the origin that owns the immutable `safetensors-cache-v1` database. It opens
that database readonly and never creates, upgrades, or writes either object store.

## Images

Structured user messages may include image parts paired with browser image sources. The owned
runtime decodes and resizes the image, patchifies RGB pixels, executes the pinned 16-layer vision
tower on WebGPU, pools and projects its output to 1,536-wide language soft tokens, and inserts those
GPU-resident rows into language prefill. Vision transformer layers are loaded and released one at a
time to bound memory. No ONNX session or CPU transformer fallback is used.

The operator, complete-layer, 16-layer tower, postprocess, tokenizer, browser preprocessing,
abort-boundary, and maximum-2,520-patch tests pass. The answer-free dolphin example completed
same-origin image encoding, chunked language prefill, and decode, accurately transcribing its
visible caption and credit line without prompt leakage. The request spent 103.6 seconds in vision
out of 110.9 seconds total. The console reports vision progress and preprocessing/encoding timing;
live mid-tower cancellation, repeated-request resource stability, maximum-geometry full generation,
and finer stage timing remain open. Multimodal requests reset prefix caches, while text-only
requests may reuse a retained common prefix.

## Context capacity

The checkpoint supports 131,072 positions, and the console currently allocates and exposes 32,768
positions. The limit includes both prompt and output:
`prompt positions + max new tokens - 1` must fit the allocated session cache. Sliding-attention
layers use 512-position circular caches; full-attention layers use the selected logical capacity.
Real allocation, generation across the 512-position boundary, unaligned prefix reuse with hybrid
chunked prefill, cancellation, post-cancellation recovery, exact-fit generation at position 8,192,
and rejection at 8,193 pass at 8K. The exact-fit request used `chunked-32`, completed in 89.1
seconds, and retained 898.5 MiB of GPU buffers. The durable certification harness checkpoints long
runs in browser storage and downloads completed JSON artifacts. At 32K, exact-fit generation
retained 1,244,172,500 bytes, emitted token `1509` (`It`) with `chunked-32`, and completed in
670.105 seconds. A follow-up reused exactly 32,767 prompt tokens and reproduced the output in
171.5 ms; 32,769 positions were rejected, cancellation checkpointed at 31,744 rows, and a clean
subsequent run completed. At 128K, exact-fit generation retained 2,452,132,052 bytes, emitted the
same token with `chunked-32`, and completed in 4,424.216 seconds. The follow-up reused exactly
131,071 prompt tokens in 1,070.9 ms; 131,073 positions were rejected, and cancellation plus clean
recovery passed at 1,024 rows. The normal console remains at the practical certified 32K tier.

## Roadmap

Work proceeds in this order: harden repeated/cancelled/maximum-geometry vision and optimize its
measured 103.6-second path;
expose role-preserving multi-turn chat and define multimodal reuse; implement the pinned audio
path; implement deterministic video frame ingestion; optimize prefill and constrained decoding;
then harden device-loss recovery and the release browser matrix. See
[PROJECT.md](PROJECT.md#execution-plan) for gates and details.

The owned and pinned runtimes now match prompt IDs, every generated ID, EOS behavior, and final
text across five greedy golden cases: a short English greeting (`Hi!`), arithmetic (`12`), Arabic
(`أهلاً بك!`), a longer primary-colors instruction, and an exact 32-token prefill boundary. The vectors live in
`src/runtime/gemma-golden.ts`. This suite exposed a full-attention defect where 512-wide heads used
256-element workgroup arrays; the arrays now compile at the profile head dimension and repeated
BOS evaluations are bit-deterministic across all owner-layer K/V caches.

Fixed prefill preserves the pinned `M=32` program: token-ID-0 padding, exact staged-SRQ QAT,
per-head RMSNorm, split-half RoPE, tiled causal/windowed GQA, shared owner K/V caches, dense signed
PLE projections, actual-last-row selection, and logical cache advancement by only the valid prompt
length. It binds decode-owned model weights rather than uploading a second model copy. All five
goldens retain exact generated IDs, EOS behavior, and text after the block-to-decode handoff.

`benchmarkGemmaGeneration()` records session load, request setup, cache reset, prefill, TTFT, every
decode step, logits readback, callback time, total latency, throughput, exact golden parity, adapter
identity, and deduplicated retained GPU-buffer bytes. In Electron 42.5.0 / Chrome 148 on Apple
Metal 3, the 19-token longer golden measured median TTFT `727.0 ms` fixed versus `379.0 ms`
sequential and median total latency `1139.4 ms` versus `612.4 ms`. At the exact 32-token boundary,
fixed measured `783.8 ms` median TTFT versus `745.8 ms` sequential; fixed had the better p95 but did
not improve median latency. It also retained 2,849 buffers / 847,810,772 bytes versus 1,640 buffers /
832,581,332 bytes. The speedup gate therefore failed and automatic routing remains sequential.
[benchmarks/full-generation-longer-instruction.electron-148.json](benchmarks/full-generation-longer-instruction.electron-148.json),
[benchmarks/full-generation-longer-instruction-sequential.electron-148.json](benchmarks/full-generation-longer-instruction-sequential.electron-148.json),
and [benchmarks/full-generation-prefill-crossover.electron-148.json](benchmarks/full-generation-prefill-crossover.electron-148.json)
retain the raw samples; no fixed-prefill speedup is claimed.

The complete path uses four input-preparation dispatches, the 276-dispatch transformer stack, one
LM-head dispatch, and two deterministic argmax dispatches: 283 compute dispatches per evaluated
token.

Model weights are intentionally not included. The immutable upstream revision, artifact hashes, architecture, tokenizer/generation identity, and QAT representation are recorded in [public/models/gemma-4-e2b/manifest.json](public/models/gemma-4-e2b/manifest.json).

The lab benchmarks owned packed-int4 layer-0 Q and fused QKV implementations at the real `1536 -> 2048/256/256` decode shape. Their weights come from verified readonly exports of the browser cache used by Buza. The operator implementation belongs to Hugging Face's [Gemma 4 WebGPU kernels](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels) runtime at commit `158f16ae0f672943ca304d59c47c8e3a264e399e`; Buza vendors and invokes a byte-identical bundle with SHA-256 `0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62`. Buza was used only as the cache-owning host and capture harness.

The source-equivalent Q-only `QatMatMul scalar_presrq` path reproduces all 2,048 captured Q values bit-for-bit and measures `0.023593 ms` GPU-timestamp median, a `25.6x` kernel-time improvement over the original owned `0.603 ms` Q baseline. This ratio is Q-only to Q-only; it is not compared against fused QKV.

The current equivalent-boundary evidence is [benchmarks/qat-qkv-layer0-huggingface-decode.chrome-149.json](benchmarks/qat-qkv-layer0-huggingface-decode.chrome-149.json). The owned combined-storage `DecodeQkvProj presrq` path matches every captured Q, K, and V value exactly, uses seven persistent GPU buffers with zero per-dispatch allocation, and measures `0.030802 ms` median. Hugging Face's pinned fused operator measures `0.01769472 ms` median in the same warm Chrome tab with the same 100-dispatch timestamp protocol. The owned path is currently `1.74x` slower, so the `1.5x` speedup gate is not met.

The owned `DecodeRmsSrq` path now reproduces the real layer-0 F32 SRQ activation and `sumA` bit-for-bit with the source 256-thread reduction order. It measures `0.015073 ms` median over 100 dispatches per timestamp sample. The composed path binds those output buffers directly into owned `DecodeQkvProj`, has no intermediate CPU readback or per-dispatch allocation, and preserves exact Q/K/V output. In an equivalent same-tab comparison using 100 ordered two-pass pairs per timestamp sample, owned measures `0.034734 ms` median and Hugging Face measures `0.02097152 ms`. The owned pair is currently `1.66x` slower, so no speedup is claimed. [benchmarks/decode-rms-qkv-layer0-owned.chrome-149.json](benchmarks/decode-rms-qkv-layer0-owned.chrome-149.json) records the full protocol and result.

The source-equivalent 128-thread `DecodeQkNormRope scalar` path reproduces all 256 captured layer-0 K values bit-for-bit after RMS normalization and split-half RoPE. In headless Chrome 149 it measures `0.003932 ms` median over 100 dispatches per timestamp sample. Owned QKV feeds its K subrange directly into this kernel, which writes normalized K to cache position 10 through the source `dstOffset` contract with no intermediate CPU readback or GPU copy; that Chrome 149 owned-only pair measures `0.037356 ms` median. [benchmarks/decode-qkv-k-norm-rope-layer0-owned.chrome-149.json](benchmarks/decode-qkv-k-norm-rope-layer0-owned.chrome-149.json) retains that environment-specific evidence.

An equivalent comparison was also run in the cache-owning Electron 42.5.0 / Chrome 148 tab using three alternating rounds, 30 samples per implementation, and 100 ordered two-pass pairs per sample. Owned measures `0.020972 ms` median versus Hugging Face's pinned `0.02228224 ms`, a `1.06x` median improvement, while owned p95 is `1.05x` slower. The result is exact but does not meet the project's `1.5x` gate and is not relabeled as Chrome 149 evidence. [benchmarks/decode-qkv-k-norm-rope-layer0-equivalent.electron-148.json](benchmarks/decode-qkv-k-norm-rope-layer0-equivalent.electron-148.json) preserves all raw samples.

The owned fixed-32-lane-subgroup `Gemma4DecodeAttentionPartial` implementation reproduces all 2,048 captured layer-0 outputs bit-for-bit. The single dispatch includes Q RMSNorm, split-half RoPE, grouped-query attention over the logical K/V cache prefix, flash partial accumulation, last-arriver merge, and output SRQ. It uses ten persistent buffers totaling 313,408 bytes and allocates nothing per dispatch. In headless Chrome 149 it measures `0.043202 ms` median and `0.056177 ms` p95 over 20 dispatches per timestamp sample; [benchmarks/decode-attention-layer0-owned.chrome-149.json](benchmarks/decode-attention-layer0-owned.chrome-149.json) records that owned-only environment.

The equivalent attention boundary was measured separately in the same cache-owning Electron 42.5.0 / Chrome 148 tab using three alternating rounds and 30 samples per implementation. Owned and pinned Hugging Face occupy the same median timer bucket (`0.00983 ms` and `0.0098304 ms`); owned p95 is `0.00983 ms` versus `0.0131072 ms`, but no meaningful speedup is claimed at this timer resolution. [benchmarks/decode-attention-layer0-equivalent.electron-148.json](benchmarks/decode-attention-layer0-equivalent.electron-148.json) retains every raw sample.

The owned fixed-32-lane `DecodeOprojNorm` implementation reproduces the pinned fused output projection boundary exactly in one dispatch: packed 4-bit `2048 -> 1536` GEMV, projection SRQ, post-attention residual/RMS normalization, pre-FFN RMS normalization and F16 SRQ, and the F32 FFN-input sum. The updated hidden state and sum match bit-for-bit, and all 1,536 F16 output bits match. Its strict fixture is derived from the cache-owning Buza host without writing to IndexedDB; the corrected source capture binds `sum2T` to the runtime's actual `g4d-ff-suma` tensor.

The layer-0 decode-critical attention block now runs from the captured hidden state through owned `DecodeRmsSrq`, direct-cache QKV projection, K RMSNorm/RoPE, weightless V RMSNorm, fused attention, and `DecodeOprojNorm` in one ordered six-pass submission. The attention output buffer binds directly into O projection, which mutates the original hidden buffer and emits the F16 MLP input and sum without an intermediate copy or readback. Q, raw K, cached normalized K/V, all 2,048 attention outputs, updated hidden, every F16 MLP-input bit, and the sum remain exact. The path uses 28 persistent buffers totaling 3,931,296 bytes with no inter-kernel GPU copies, CPU readbacks, or per-dispatch allocations. No composed-block speed claim is made until equivalent GPU timestamp evidence is collected.

The remaining layer-0 MLP/PLE boundary is also owned and composed through four ordered dispatches: `DecodeGateUpNormPresrq`, `DecodeDownNormAddFused`, `DecodePleGateCodes`, and `DecodePleProjNormCodes`. Joined to the attention block, the complete ten-dispatch layer aliases the O-projection hidden and pre-MLP outputs directly into the MLP/PLE resources. Its validation harness uses 55 GPU buffers totaling 19,026,168 bytes, including final readback resources, with no copies, readbacks, or allocations between kernels. Final hidden and next-layer sum are bit-exact; the next-layer input has no nonzero mismatch and only 38 signed-zero differences between Chrome 148 and 149.

All-layer loading now starts from an owned readonly IndexedDB reader for the existing `safetensors-cache-v1` artifact. It enumerates the database before opening its reported version, validates the pinned source URL, file size, data offset, and 2,780-tensor header contract, then reads exact compound-key tensor ranges without creating, upgrading, or writing either object store. This preserves the cache ownership and provenance constraints while avoiding another full model download.

The model planner validates all 35 layers directly against that pinned header: 1,590 canonical tensors totaling 675,998,518 bytes across sliding/full attention and int4/int2 MLP profiles. Layer loading batches each plan into one readonly transaction, verifies every payload length, and retains a SHA-256 per tensor while staging only one layer at a time.

The owned MLP block now selects either the 6,144-wide int4 or 12,288-wide int2 pipelines from the materialized layer profile. Every materialized layer also receives its own pinned-arithmetic GELU lookup tables derived from that layer's gate and PLE output scales; reusing the captured layer-0 tables was the final model-wide parity defect. The layer-15 int2 path uses 96 packed words per gate/up row, zero point 2, 3,072 gate/up workgroups, and four activation vectors per down-projection word while retaining the four-dispatch shared-buffer plan. A canonical-shape synthetic GPU test compiles, binds, and executes the complete profile, checks one gate/up row as exact F16 code 9, and isolates one nonzero down row.

Canonical layer-0 tensors now materialize into every model-owned resource consumed by the ten-dispatch GPU plan. The immutable-range verifier checks 19 packed-weight, scale, and fused-norm resources against the exact captures; all 18,576,466 source bytes map exactly. In particular, U8 QAT bytes are reinterpreted directly, while signed I8 PLE weights are rebased to the unsigned zero-point-128 codes expected by WGSL. Run `npm run verify:materializer` to repeat this range-scoped check without downloading the full model.

The earlier [benchmarks/qat-linear-layer0-qproj-buza-decode.chrome-149.json](benchmarks/qat-linear-layer0-qproj-buza-decode.chrome-149.json) records the original standalone-Q baseline and the then non-equivalent fused timing evidence.

The earlier [benchmarks/qat-linear-layer0-qproj-real-activation.chrome-149.json](benchmarks/qat-linear-layer0-qproj-real-activation.chrome-149.json) remains the prior Transformers-derived real-activation baseline, and [benchmarks/qat-linear-layer0-qproj-real.chrome-149.json](benchmarks/qat-linear-layer0-qproj-real.chrome-149.json) remains the real-weight/synthetic-activation projection-core measurement.

Reference export and existing browser-cache reuse are documented in [docs/REFERENCE.md](docs/REFERENCE.md) and [docs/CACHE_REUSE.md](docs/CACHE_REUSE.md).

## Constrained decoding

Generation supports tokenizer-aware output constraints as a first-class API alongside the existing sampling controls:

```ts
type GenerationConstraint =
	| { type: "regex"; pattern: string }
	| { type: "json"; maxDepth?: number; whitespace?: "none" | "compact" | "any" }
	| {
			type: "json-schema";
			schema: object;
			maxDepth?: number;
			whitespace?: "none" | "compact" | "any";
		};
```

JSON Schema is the preferred contract for Buza decision payloads. Regex constraints target regular output formats and deliberately reject assertions and backreferences. Nested JSON uses a bounded grammar rather than pretending an unbounded JSON language is regular. The supported schema subset includes `type`, `const`, `enum`, `oneOf`, `anyOf`, closed objects whose declared properties are all required, and bounded homogeneous arrays. Unknown keywords, references, optional/open object properties, and other unsupported constructs fail configuration explicitly.

Constraints operate on the exact tokenizer's UTF-8 token bytes, including partial multi-byte characters across token boundaries. A token trie and minimized DFA determine legal token IDs for each state. The current correctness-first path masks the resident CPU logits before top-k/min-p/typical-p/top-p filtering and sampling; GPU candidate masking remains a performance milestone. EOS and configured stop tokens are legal only in an accepting state, dead ends are explicit errors, and regex, JSON, and AJV schema checks validate final output independently.

Constrained decoding does not reduce the cost of Gemma's transformer pass by itself. Its expected end-to-end gains come from preventing malformed output and retries, stopping at the first complete accepted payload, reducing sampling/readback work, and producing fewer unnecessary tokens. Performance claims require measured constrained and unconstrained runs with the same prompt and output contract.

Focused browser tests cover trie pruning, full-match regex behavior, split-token UTF-8, invalid syntax, bounded JSON, the closed schema subset, unsupported constructs, and masking. Live cached-model gates produced exact constrained `Hi!` and `{"ok":true}` outputs with streaming/JSON parity and no WebGPU validation or internal errors.
