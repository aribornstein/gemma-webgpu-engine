#!/usr/bin/env python3
"""Export deterministic Gemma 4 golden vectors from an already-local snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c"
WEIGHTS_SHA256 = "efab429012b97ab986c4d4838a46ff3ad95d618b42ce514771ca40fadc76a9a4"
DEFAULT_LAYERS = (0, 15, 34)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("reference-output"), type=Path)
    parser.add_argument("--prompt", default="Write one short greeting.")
    parser.add_argument("--max-new-tokens", default=8, type=int)
    parser.add_argument("--layers", default=DEFAULT_LAYERS, nargs="+", type=int)
    parser.add_argument("--verify-weights", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_snapshot(model_dir: Path, verify_weights: bool) -> None:
    required = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Missing pinned local artifacts: {', '.join(missing)}")
    if verify_weights:
        actual = sha256(model_dir / "model.safetensors")
        if actual != WEIGHTS_SHA256:
            raise SystemExit(f"Weight SHA-256 mismatch: {actual}")


def language_layers(model: torch.nn.Module) -> Any:
    current: Any = model
    for name in ("model", "language_model", "layers"):
        if not hasattr(current, name):
            raise SystemExit(f"Expected model path model.language_model.layers; missing {name}")
        current = getattr(current, name)
    return current


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor hook value, got {type(value).__name__}")
    return value.detach().to(dtype=torch.float32, device="cpu").numpy()


def main() -> None:
    args = parse_args()
    require_snapshot(args.model_dir, args.verify_weights)
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()

    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    selected_layers = tuple(args.layers)
    captures: dict[str, np.ndarray] = {}
    handles = []
    layers = language_layers(model)
    for layer_index in selected_layers:
        if layer_index < 0 or layer_index >= len(layers):
            raise SystemExit(f"Layer index {layer_index} is outside 0..{len(layers) - 1}")

        def capture_input(_module: Any, values: tuple[Any, ...], index: int = layer_index) -> None:
            captures[f"layer_{index}_input"] = to_numpy(values)

        def capture_output(_module: Any, _values: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
            captures[f"layer_{index}_output"] = to_numpy(output)

        handles.append(layers[layer_index].register_forward_pre_hook(capture_input))
        handles.append(layers[layer_index].register_forward_hook(capture_output))

    with torch.inference_mode():
        result = model(**encoded, use_cache=False, return_dict=True)
        last_token_logits = result.logits[0, -1].to(dtype=torch.float32, device="cpu")
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=[1, 106, 50],
            pad_token_id=0,
        )[0, encoded["input_ids"].shape[1]:]
    for handle in handles:
        handle.remove()

    captures["final_last_token_logits"] = last_token_logits.numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_file = args.output_dir / "golden_tensors.npz"
    np.savez_compressed(tensor_file, **captures)
    metadata = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "revision": REVISION,
        "weightsSha256": WEIGHTS_SHA256,
        "prompt": args.prompt,
        "messages": messages,
        "renderedPrompt": rendered,
        "inputIds": encoded["input_ids"][0].tolist(),
        "attentionMask": encoded["attention_mask"][0].tolist(),
        "selectedLayers": list(selected_layers),
        "tensorShapes": {name: list(value.shape) for name, value in captures.items()},
        "finalLogitsArgmax": int(last_token_logits.argmax().item()),
        "greedyTokenIds": generated.tolist(),
        "greedyText": tokenizer.decode(generated, skip_special_tokens=False),
        "generationEosTokenIds": [1, 106, 50],
        "tensorFile": tensor_file.name,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
    }
    (args.output_dir / "golden.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metadata": "golden.json", "tensors": tensor_file.name}, indent=2))


if __name__ == "__main__":
    main()