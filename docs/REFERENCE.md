# Trusted reference export

`tools/export_reference.py` exports the exact chat-template rendering, tokenizer IDs, selected transformer layer inputs and outputs, final last-token logits, and greedy token IDs. Tensor data is stored as non-pickle NPZ and metadata as JSON.

The exporter is deliberately offline-only. `--model-dir` must point to an already-local copy of snapshot `9fcec64df66cb1e4d972fc5cdc142afb25b2362c`; Transformers receives `local_files_only=True` and cannot initiate the 2.46 GB weight download. Pass `--verify-weights` to verify `model.safetensors` against SHA-256 `efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4` before loading.

Create a dedicated Python environment with a Transformers build that supports `Gemma4ForConditionalGeneration`, PyTorch, and NumPy, then run:

```bash
python tools/export_reference.py \
  --model-dir /path/to/pinned/snapshot \
  --output-dir reference-output \
  --verify-weights
```

The exported JSON records exact Python, PyTorch, Transformers, and NumPy versions. Lock those versions after the first successful export and retain the generated files with correctness results. The default layers are 0, 15, and 34, covering the first 4-bit MLP region, the first default 2-bit MLP region, and the final layer.

## Minimal layer-0 reference

`tools/export_layer0_reference.py` avoids loading a full model. It consumes the seven cached tensors required to compute `input_layernorm(embed_tokens(input_ids))` and the layer-0 Q projection. It uses Transformers' own `QuantizedEmbedding`, `Gemma4RMSNorm`, `QuantizedLinear`, and `apply_srq` implementations, with the exact tokenizer and chat template. No network access occurs during the export.

The validated environment uses Python 3.12.13, PyTorch 2.8.0, NumPy 2.5.1, safetensors 0.8.0, and Transformers commit `63f32a8782cb70da3365acab16f2b67947737985`. Run:

```bash
.venv/bin/python tools/export_layer0_reference.py \
  --fixture reference-local/gemma-4-e2b-layer0-reference.safetensors \
  --tokenizer-dir reference-local/gemma-4-e2b-tokenizer \
  --output-dir reference-output/layer0
```

The default prompt renders to 14 exact token IDs. The selected final prompt token produces pre-SRQ input, SRQ-rounded input, and post-SRQ output vectors in `layer0-q-proj-golden.safetensors`. The published 20,704-byte golden fixture has SHA-256 `e9ef6a3b477cda98572611840f6a87965a9460221f4bc2b5473f535b50b876f2`; its WebGPU output matches all 2,048 trusted values exactly.