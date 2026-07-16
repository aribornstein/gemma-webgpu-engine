#!/usr/bin/env python3
"""Export the exact layer-0 fused decode-attention boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

OPERATOR = "Gemma4DecodeAttentionPartial"
UPSTREAM_COMMIT = "158f16ae0f672943ca304d59c47c8e3a264e399e"
BUNDLE_SHA256 = "0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62"
CAPTURE_SHA256 = "fa4d670c13f3f7e1d271040b994aefb85106fe0b0343646907fb54cc9b907f2b"
EXPECTED_ATTENTION = {
    "operator": OPERATOR,
    "qHeads": 8,
    "kvHeads": 1,
    "headDim": 256,
    "keyLen": 11,
    "qOffset": 10,
    "window": 512,
    "epsilon": 1e-6,
    "scale": 1,
    "outputQuantScale": 0.03026575781404972,
    "ropeTable": "sliding",
}
TENSORS = {
    "q": ([1, 8, 256], "ab887bc85137320455da9e3a9b5b8e121bb19dced68326cfa1434f82b951482b"),
    "qNormWeight": ([256], "9bd79dd6adec377becdb3b6438691fa38b20e17c40826eb5698e16086093b347"),
    "ropeCos": ([1, 128], "0a8d34ce643de6612a8725b36cb6599f3de2bb956fbd3c96f7577014c9707141"),
    "ropeSin": ([1, 128], "078b5542f3148c0eeb381bd8be0bec9b2770f366f0466093c984e659807b5132"),
    "keyCache": ([11, 1, 256], "7bbcbafa49c38c6897649de552926c05d4737120dda4c81bbb4b3f4a1d242fd8"),
    "valueCache": ([11, 1, 256], "87f0773acac7e9b6ab2814d3fd4771cf6c803223546379bf521f097e21fe0f56"),
    "output": ([1, 8, 256], "3273707bd3456f41e904caa6aeaad622db0abda6c972840c3d8d69310bd913eb"),
}
OUTPUT_NAMES = {
    "q": "q",
    "qNormWeight": "q_norm_weight",
    "ropeCos": "cosine",
    "ropeSin": "sine",
    "keyCache": "key_cache",
    "valueCache": "value_cache",
    "output": "output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def captured_tensor(capture: dict[str, object], name: str) -> np.ndarray:
    shape, expected_sha256 = TENSORS[name]
    tensors = capture.get("tensors")
    if not isinstance(tensors, dict) or not isinstance(tensors.get(name), dict):
        raise SystemExit(f"Capture is missing tensor {name}")
    tensor = tensors[name]
    if tensor.get("dtype") != "float32" or tensor.get("shape") != shape:
        raise SystemExit(f"Capture metadata mismatch for {name}")
    value = np.asarray(tensor.get("values"), dtype=np.float32).reshape(shape)
    actual_sha256 = sha256_bytes(f32_bytes(value))
    if tensor.get("sha256") != actual_sha256 or actual_sha256 != expected_sha256:
        raise SystemExit(f"Capture tensor hash mismatch for {name}: {actual_sha256}")
    return value


def main() -> None:
    args = parse_args()
    if sha256_file(args.capture) != CAPTURE_SHA256:
        raise SystemExit("Decode attention source capture hash mismatch")
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    if (
        capture.get("source") != {
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        }
        or capture.get("attention") != EXPECTED_ATTENTION
        or capture.get("tokens") != {
            "inputIds": [2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107],
            "decodeInputToken": 9259,
            "position": 10,
        }
    ):
        raise SystemExit("Decode attention capture identity or metadata mismatch")

    arrays = {name: captured_tensor(capture, name) for name in TENSORS}
    output_tensors = {
        OUTPUT_NAMES[name]: torch.from_numpy(value.copy())
        for name, value in arrays.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "decode-attention-layer0.safetensors"
    save_file(output_tensors, tensor_path)

    output_metadata = {
        "schemaVersion": 1,
        "sourceCapture": {
            "file": args.capture.name,
            "sha256": CAPTURE_SHA256,
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        },
        "operator": OPERATOR,
        "tokens": capture["tokens"],
        "attention": EXPECTED_ATTENTION,
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
    metadata_path = args.output_dir / "decode-attention-layer0.json"
    metadata_path.write_text(
        json.dumps(output_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_metadata, indent=2))


if __name__ == "__main__":
    main()
