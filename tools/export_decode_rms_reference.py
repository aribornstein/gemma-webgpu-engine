#!/usr/bin/env python3
"""Export the exact layer-0 decode RMS/SRQ boundary from verified local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

EMBEDDING = "model.language_model.embed_tokens"
INPUT_NORM = "model.language_model.layers.0.input_layernorm"
Q_PROJ = "model.language_model.layers.0.self_attn.q_proj"
HIDDEN_SIZE = 1536
EMBED_BITS = 2
EMBED_ZERO_POINT = 2
RMS_NORM_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float32).tobytes()


def main() -> None:
    args = parse_args()
    tensors = load_file(args.fixture, device="cpu")
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    token_id = int(capture["tokens"]["decodeInputToken"])

    packed_row = tensors[f"{EMBEDDING}.embedding_quantized"][token_id].numpy()
    row_scale = np.float32(tensors[f"{EMBEDDING}.embedding_scale"][token_id, 0])
    norm_weight = (
        tensors[f"{INPUT_NORM}.weight"].float().numpy().astype(np.float32)
    )
    input_scale = np.float32(tensors[f"{Q_PROJ}.input_activation_scale"])
    embed_scale = np.float32(math.sqrt(HIDDEN_SIZE))

    hidden = np.empty(HIDDEN_SIZE, dtype=np.float32)
    values_per_byte = 8 // EMBED_BITS
    mask = (1 << EMBED_BITS) - 1
    for byte_index, packed in enumerate(packed_row):
        for value_index in range(values_per_byte):
            code = np.float32((int(packed) >> (value_index * EMBED_BITS)) & mask)
            hidden[byte_index * values_per_byte + value_index] = np.float32(
                np.float32(embed_scale * row_scale)
                * np.float32(code - np.float32(EMBED_ZERO_POINT))
            )

    mean_square = np.mean(hidden * hidden, dtype=np.float32)
    rms_scale = np.float32(
        1.0 / np.sqrt(np.float32(mean_square + np.float32(RMS_NORM_EPS)))
    )
    normalized = np.float32(hidden * rms_scale * norm_weight)
    computed_output = np.float32(
        np.clip(np.rint(np.float32(normalized / input_scale)), -128, 127)
        * input_scale
    )
    computed_sum = np.array([np.sum(computed_output, dtype=np.float32)], dtype=np.float32)

    captured_output = np.asarray(capture["tensors"]["input"]["values"], dtype=np.float32)
    captured_sum = np.asarray(capture["tensors"]["sumA"]["values"], dtype=np.float32)
    if not np.array_equal(computed_output, captured_output):
        changed = int(np.count_nonzero(computed_output != captured_output))
        raise SystemExit(f"Derived RMS output differs from capture at {changed} values")
    if not np.array_equal(computed_sum, captured_sum):
        raise SystemExit(
            f"Derived RMS sum {computed_sum[0]} differs from capture {captured_sum[0]}"
        )

    output_tensors = {
        "hidden": torch.from_numpy(hidden.copy()),
        "weight": torch.from_numpy(norm_weight.copy()),
        "output": torch.from_numpy(captured_output.copy()),
        "sum_a": torch.from_numpy(captured_sum.copy()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "decode-rms-srq-layer0.safetensors"
    save_file(output_tensors, tensor_path)

    metadata = {
        "schemaVersion": 1,
        "sourceFixture": {
            "file": args.fixture.name,
            "sha256": sha256_file(args.fixture),
        },
        "sourceCapture": {
            "file": args.capture.name,
            "sha256": sha256_file(args.capture),
            "upstreamCommit": capture["source"]["upstreamCommit"],
            "bundleSha256": capture["source"]["bundleSha256"],
        },
        "operator": "com.xenova.gemma4.DecodeRmsSrq",
        "tokenId": token_id,
        "hiddenSize": HIDDEN_SIZE,
        "epsilon": RMS_NORM_EPS,
        "inputScale": float(input_scale),
        "tensorFile": tensor_path.name,
        "tensorFileSha256": sha256_file(tensor_path),
        "tensors": {
            name: {
                "dtype": "float32",
                "shape": list(value.shape),
                "sha256": sha256_bytes(f32_bytes(value.numpy())),
            }
            for name, value in output_tensors.items()
        },
    }
    metadata_path = args.output_dir / "decode-rms-srq-layer0.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()