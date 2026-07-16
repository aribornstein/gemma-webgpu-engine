#!/usr/bin/env python3
"""Export an exact layer-0 Gemma 4 Q-projection vector from a minimal fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear, apply_srq
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
REVISION = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c"
EMBEDDING = "model.language_model.embed_tokens"
INPUT_NORM = "model.language_model.layers.0.input_layernorm"
Q_PROJ = "model.language_model.layers.0.self_attn.q_proj"
VOCAB_SIZE = 262144
HIDDEN_SIZE = 1536
Q_OUTPUT_SIZE = 2048
RMS_NORM_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("reference-output/layer0"), type=Path)
    parser.add_argument("--prompt", default="Write one short greeting.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(fixture: Path, tokenizer_dir: Path) -> None:
    required = [
        fixture,
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
        tokenizer_dir / "chat_template.jinja",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing local reference artifacts: {', '.join(missing)}")


def require_tensor(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    tensor = tensors.get(name)
    if tensor is None:
        available = ", ".join(sorted(tensors))
        raise SystemExit(f"Fixture is missing {name}. Available tensors: {available}")
    return tensor


def frozen_parameter(value: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(value, requires_grad=False)


def build_embedding(tensors: dict[str, torch.Tensor]) -> QuantizedEmbedding:
    module = QuantizedEmbedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=HIDDEN_SIZE,
        output_dtype=torch.bfloat16,
        embed_scale=math.sqrt(HIDDEN_SIZE),
        num_bits=2,
    )
    module.embedding_quantized = frozen_parameter(
        require_tensor(tensors, f"{EMBEDDING}.embedding_quantized"),
    )
    module.embedding_scale = frozen_parameter(
        require_tensor(tensors, f"{EMBEDDING}.embedding_scale"),
    )
    return module.eval()


def build_input_norm(tensors: dict[str, torch.Tensor]) -> Gemma4RMSNorm:
    module = Gemma4RMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS)
    module.weight = frozen_parameter(require_tensor(tensors, f"{INPUT_NORM}.weight"))
    return module.eval()


def build_q_projection(tensors: dict[str, torch.Tensor]) -> QuantizedLinear:
    module = QuantizedLinear(HIDDEN_SIZE, Q_OUTPUT_SIZE, bias=False, num_bits=4)
    module.weight = frozen_parameter(require_tensor(tensors, f"{Q_PROJ}.weight"))
    module.weight_scale = frozen_parameter(require_tensor(tensors, f"{Q_PROJ}.weight_scale"))
    module.input_activation_scale = frozen_parameter(
        require_tensor(tensors, f"{Q_PROJ}.input_activation_scale"),
    )
    module.output_activation_scale = frozen_parameter(
        require_tensor(tensors, f"{Q_PROJ}.output_activation_scale"),
    )
    return module.eval()


def tensor_metadata(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in tensors.items()
    }


def main() -> None:
    args = parse_args()
    require_files(args.fixture, args.tokenizer_dir)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    messages = [{"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    tensors = load_file(args.fixture, device="cpu")
    embedding = build_embedding(tensors)
    input_norm = build_input_norm(tensors)
    q_projection = build_q_projection(tensors)

    with torch.inference_mode():
        hidden_states = embedding(encoded["input_ids"])
        q_input = input_norm(hidden_states)
        q_input_srq = apply_srq(q_input, q_projection.input_activation_scale)
        q_output = q_projection(q_input)

    last_token = encoded["input_ids"].shape[1] - 1
    golden_tensors = {
        "q_input": q_input[0, last_token].float().contiguous(),
        "q_input_srq": q_input_srq[0, last_token].float().contiguous(),
        "q_output": q_output[0, last_token].float().contiguous(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_file = args.output_dir / "layer0-q-proj-golden.safetensors"
    save_file(golden_tensors, tensor_file)
    np.save(args.output_dir / "input_ids.npy", encoded["input_ids"].numpy(), allow_pickle=False)

    metadata = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "revision": REVISION,
        "fixture": {
            "file": args.fixture.name,
            "sha256": sha256(args.fixture),
            "tensors": tensor_metadata(tensors),
        },
        "prompt": args.prompt,
        "messages": messages,
        "renderedPrompt": rendered,
        "inputIds": encoded["input_ids"][0].tolist(),
        "selectedTokenIndex": last_token,
        "selectedTokenId": int(encoded["input_ids"][0, last_token]),
        "goldenTensorFile": tensor_file.name,
        "goldenTensorSha256": sha256(tensor_file),
        "goldenTensors": tensor_metadata(golden_tensors),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "modelDtype": "bfloat16",
            "exportDtype": "float32",
        },
    }
    metadata_file = args.output_dir / "layer0-q-proj-golden.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "metadata": str(metadata_file),
        "tensors": str(tensor_file),
        "fixtureSha256": metadata["fixture"]["sha256"],
        "goldenSha256": metadata["goldenTensorSha256"],
    }, indent=2))


if __name__ == "__main__":
    main()