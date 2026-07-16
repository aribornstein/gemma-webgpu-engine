#!/usr/bin/env python3
"""Export the exact layer-0 DecodeOprojNorm boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

UPSTREAM_COMMIT = "158f16ae0f672943ca304d59c47c8e3a264e399e"
BUNDLE_SHA256 = "0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62"
CAPTURE_METADATA_SHA256 = "1b81dd537bc0418ce74d93ee5dcf8c0b5d4b70c4c8621bb2ec3ae15c3d0dacdf"
CAPTURE_TENSORS_SHA256 = "b5eff21d1af5f8826cd00a2a01d0830462fad78505a803c50ef1ca12b8e2ac52"
EXPECTED_OPERATOR = {
    "name": "DecodeOprojNorm",
    "bits": 4,
    "inFeatures": 2048,
    "outFeatures": 1536,
    "outputScale": 0.21056734025478363,
    "epsilon": 1e-6,
    "inScale2": 0.9406865835189819,
    "rows": 0,
}
EXPECTED_TOKENS = {
    "inputIds": [2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107],
    "decodeInputToken": 9259,
    "position": 10,
}
TENSORS = {
    "attention": ("attention", "float32", "Float32Array", [1, 2048], "3273707bd3456f41e904caa6aeaad622db0abda6c972840c3d8d69310bd913eb"),
    "packedWeights": ("packed_weights", "uint32", "Uint32Array", [1536, 256], "a6bccc46f469f441df077de2d07b4b07cc3b1b14358384938a139562eb76f6e2"),
    "rowScales": ("row_scales", "float32", "Float32Array", [1536], "f2070a12a9ecf0c3a8874cb19c345253461edb517cae7f062c795cbee7b75e8e"),
    "hiddenBefore": ("hidden_before", "float32", "Float32Array", [1, 1536], "4b7c92982f60cc985f150bdbb22a28a8e276bf516edd2b6a24c284fb810ed6a2"),
    "normWeights": ("norm_weights", "float32", "Float32Array", [2, 1536], "0ed24ca107f0365f5e72538f1042c566b7c9cf73fb418c5183c8aa510ae122d8"),
    "hiddenAfter": ("hidden_after", "float32", "Float32Array", [1, 1536], "6bfbb8d43ec4bee447817403dd693194d438b3a3b9b390eca438922d77ac7e6a"),
    "ffnInput": ("ffn_input", "float16", "Uint16Array", [1, 1536], "b5be5add660db27528dc03e0ce096b435db77f97a2429aeadc56252ba637e487"),
    "ffnInputSum": ("ffn_input_sum", "float32", "Float32Array", [1], "7e7adb267c616e101e6817e8426bc641289047bc2624e955abb90aeb25ee700f"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-metadata", required=True, type=Path)
    parser.add_argument("--capture-tensors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor) -> bytes:
    return np.ascontiguousarray(tensor.numpy()).tobytes()


def main() -> None:
    args = parse_args()
    if sha256_file(args.capture_metadata) != CAPTURE_METADATA_SHA256:
        raise SystemExit("DecodeOprojNorm capture metadata hash mismatch")
    if sha256_file(args.capture_tensors) != CAPTURE_TENSORS_SHA256:
        raise SystemExit("DecodeOprojNorm capture tensor container hash mismatch")
    capture = json.loads(args.capture_metadata.read_text(encoding="utf-8"))
    if (
        capture.get("schemaVersion") != 1
        or capture.get("source") != {
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        }
        or capture.get("tokens") != EXPECTED_TOKENS
        or capture.get("operator") != EXPECTED_OPERATOR
        or capture.get("tensorFileSha256") != CAPTURE_TENSORS_SHA256
    ):
        raise SystemExit("DecodeOprojNorm capture identity or metadata mismatch")

    source_tensors = load_file(args.capture_tensors, device="cpu")
    output_tensors = {}
    output_metadata = {}
    for capture_name, (tensor_name, dtype, array_type, shape, expected_hash) in TENSORS.items():
        metadata = capture.get("tensors", {}).get(capture_name)
        if metadata != {
            "dtype": dtype,
            "arrayType": array_type,
            "shape": shape,
            "sha256": expected_hash,
        }:
            raise SystemExit(f"DecodeOprojNorm metadata mismatch for {capture_name}")
        tensor = source_tensors.get(tensor_name)
        if tensor is None or list(tensor.shape) != shape:
            raise SystemExit(f"DecodeOprojNorm tensor mismatch for {tensor_name}")
        actual_hash = hashlib.sha256(tensor_bytes(tensor)).hexdigest()
        if actual_hash != expected_hash:
            raise SystemExit(f"DecodeOprojNorm payload hash mismatch for {tensor_name}")
        output_tensors[tensor_name] = tensor.contiguous()
        output_metadata[tensor_name] = {
            "dtype": dtype,
            "shape": shape,
            "sha256": actual_hash,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "decode-oproj-norm-layer0.safetensors"
    save_file(output_tensors, tensor_path)
    metadata = {
        "schemaVersion": 1,
        "sourceCapture": {
            "metadataFile": args.capture_metadata.name,
            "metadataSha256": CAPTURE_METADATA_SHA256,
            "tensorFile": args.capture_tensors.name,
            "tensorSha256": CAPTURE_TENSORS_SHA256,
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        },
        "operator": EXPECTED_OPERATOR,
        "tokens": EXPECTED_TOKENS,
        "tensorFile": tensor_path.name,
        "tensorFileSha256": sha256_file(tensor_path),
        "tensors": output_metadata,
    }
    metadata_path = args.output_dir / "decode-oproj-norm-layer0.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()