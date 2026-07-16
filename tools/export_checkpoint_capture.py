#!/usr/bin/env python3
"""Validate and export a browser-generated checkpoint capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

UPSTREAM_COMMIT = "158f16ae0f672943ca304d59c47c8e3a264e399e"
BUNDLE_SHA256 = "0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62"
DTYPES = {
    "float16": (np.dtype("float16"), "Uint16Array"),
    "float32": (np.dtype("float32"), "Float32Array"),
    "uint8": (np.dtype("uint8"), "Uint8Array"),
    "uint32": (np.dtype("uint32"), "Uint32Array"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-metadata", required=True, type=Path)
    parser.add_argument("--capture-tensors", required=True, type=Path)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-tensors-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    metadata_sha256 = sha256_file(args.capture_metadata)
    tensors_sha256 = sha256_file(args.capture_tensors)
    if metadata_sha256 != args.expected_metadata_sha256:
        raise SystemExit(f"capture metadata hash mismatch: {metadata_sha256}")
    if tensors_sha256 != args.expected_tensors_sha256:
        raise SystemExit(f"capture tensor hash mismatch: {tensors_sha256}")

    capture = json.loads(args.capture_metadata.read_text(encoding="utf-8"))
    if capture.get("schemaVersion") != 1:
        raise SystemExit("unsupported checkpoint capture schema")
    if capture.get("source") != {
        "upstreamCommit": UPSTREAM_COMMIT,
        "bundleSha256": BUNDLE_SHA256,
    }:
        raise SystemExit("checkpoint capture source mismatch")
    if capture.get("tensorFileSha256") != tensors_sha256:
        raise SystemExit("checkpoint capture does not pin its tensor container")

    source_tensors = load_file(args.capture_tensors)
    tensor_metadata = capture.get("tensors")
    if not isinstance(tensor_metadata, dict) or set(source_tensors) != set(tensor_metadata):
        raise SystemExit("checkpoint tensor names do not match metadata")

    output_tensors: dict[str, np.ndarray] = {}
    output_metadata: dict[str, dict[str, object]] = {}
    for name, metadata in tensor_metadata.items():
        if not isinstance(metadata, dict):
            raise SystemExit(f"invalid metadata for {name}")
        dtype_name = metadata.get("dtype")
        expected_dtype = DTYPES.get(dtype_name)
        if expected_dtype is None:
            raise SystemExit(f"unsupported dtype for {name}: {dtype_name}")
        dtype, array_type = expected_dtype
        tensor = np.ascontiguousarray(source_tensors[name])
        shape = metadata.get("shape")
        expected_hash = metadata.get("sha256")
        if (
            tensor.dtype != dtype
            or metadata.get("arrayType") != array_type
            or list(tensor.shape) != shape
            or not isinstance(expected_hash, str)
        ):
            raise SystemExit(f"checkpoint tensor contract mismatch for {name}")
        actual_hash = sha256_bytes(tensor.tobytes())
        if actual_hash != expected_hash:
            raise SystemExit(f"checkpoint tensor payload mismatch for {name}: {actual_hash}")
        output_tensors[name] = tensor
        output_metadata[name] = {
            "dtype": dtype_name,
            "arrayType": array_type,
            "shape": shape,
            "sha256": actual_hash,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / f"{args.output_stem}.safetensors"
    save_file(output_tensors, tensor_path)
    output = {
        "schemaVersion": 1,
        "sourceCapture": {
            "metadataFile": args.capture_metadata.name,
            "metadataSha256": metadata_sha256,
            "tensorFile": args.capture_tensors.name,
            "tensorSha256": tensors_sha256,
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        },
        "tokens": capture.get("tokens"),
        "steps": capture.get("steps"),
        "operators": capture.get("operators"),
        "tensorFile": tensor_path.name,
        "tensorFileSha256": sha256_file(tensor_path),
        "tensors": output_metadata,
    }
    metadata_path = args.output_dir / f"{args.output_stem}.json"
    metadata_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()