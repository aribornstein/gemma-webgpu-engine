#!/usr/bin/env python3
"""Export the exact layer-0 decode K RMSNorm/RoPE boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

OPERATOR = "com.xenova.gemma4.DecodeQkNormRope"
UPSTREAM_COMMIT = "158f16ae0f672943ca304d59c47c8e3a264e399e"
BUNDLE_SHA256 = "0234c0e866bfaa9623e938a7cfa7f5740cca22532cc1112dd4e8915b97f78d62"
CAPTURE_SHA256 = "adc161c83071906c53ecf521d4c9fa140d01f729e5ecb6017d06860592306106"
HEAD_DIM = 256
HALF_DIM = HEAD_DIM // 2
WORKGROUP_SIZE = 128
EPSILON_VALUE = 1e-6
EPSILON = np.float32(EPSILON_VALUE)
MAX_CPU_ABSOLUTE_ERROR = np.float32(2.0**-24)
MAX_CPU_ULP_ERROR = 30
TENSOR_SHA256 = {
    "k": "d7fc62c5b4f34fbcb6ec4ac1a73d1dfa7c20ed53683ac3bb86e263b246e61bb8",
    "kNormWeight": "3945bd6a02a7ff0e9dc52f6008c989907a7e91e6a8ded0de5c588fed739a8970",
    "ropeCos": "0a8d34ce643de6612a8725b36cb6599f3de2bb956fbd3c96f7577014c9707141",
    "ropeSin": "078b5542f3148c0eeb381bd8be0bec9b2770f366f0466093c984e659807b5132",
    "kNormRopeOutput": "370fb08cb473409f416d30a7df22b4cb2752ab0c2703f563fab5d2fcaeff7d94",
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


def ordered_f32(value: np.ndarray) -> np.ndarray:
    bits = value.view(np.int32).astype(np.int64)
    return np.where(bits < 0, 0x80000000 - bits, bits)


def captured_tensor(
    capture: dict[str, object],
    name: str,
    shape: list[int],
) -> np.ndarray:
    tensors = capture.get("tensors")
    if not isinstance(tensors, dict) or not isinstance(tensors.get(name), dict):
        raise SystemExit(f"Capture is missing tensor {name}")
    tensor = tensors[name]
    if tensor.get("dtype") != "float32" or tensor.get("shape") != shape:
        raise SystemExit(f"Capture metadata mismatch for {name}")
    value = np.asarray(tensor.get("values"), dtype=np.float32).reshape(shape)
    actual_sha256 = sha256_bytes(f32_bytes(value))
    if tensor.get("sha256") != actual_sha256 or actual_sha256 != TENSOR_SHA256[name]:
        raise SystemExit(f"Capture tensor hash mismatch for {name}: {actual_sha256}")
    return value


def exact_norm_rope(
    input_value: np.ndarray,
    weight: np.ndarray,
    cosine: np.ndarray,
    sine: np.ndarray,
) -> np.ndarray:
    x = input_value.reshape(HEAD_DIM)
    w = weight.reshape(HEAD_DIM)
    cos = cosine.reshape(HALF_DIM)
    sin = sine.reshape(HALF_DIM)

    reduction = np.zeros(WORKGROUP_SIZE, dtype=np.float32)
    for thread in range(WORKGROUP_SIZE):
        first = np.float32(x[thread] * x[thread])
        second = np.float32(x[thread + WORKGROUP_SIZE] * x[thread + WORKGROUP_SIZE])
        reduction[thread] = np.float32(first + second)
    stride = WORKGROUP_SIZE // 2
    while stride > 0:
        for thread in range(stride):
            reduction[thread] = np.float32(
                reduction[thread] + reduction[thread + stride]
            )
        stride //= 2

    mean_square = np.float32(reduction[0] / np.float32(HEAD_DIM))
    scale = np.float32(
        np.float32(1.0) / np.sqrt(np.float32(mean_square + EPSILON))
    )
    output = np.empty(HEAD_DIM, dtype=np.float32)
    for index in range(HALF_DIM):
        n0 = np.float32(np.float32(x[index] * scale) * w[index])
        n1 = np.float32(
            np.float32(x[index + HALF_DIM] * scale) * w[index + HALF_DIM]
        )
        output[index] = np.float32(
            np.float32(n0 * cos[index]) - np.float32(n1 * sin[index])
        )
        output[index + HALF_DIM] = np.float32(
            np.float32(n1 * cos[index]) + np.float32(n0 * sin[index])
        )
    return output


def main() -> None:
    args = parse_args()
    if sha256_file(args.capture) != CAPTURE_SHA256:
        raise SystemExit("DecodeQkNormRope source capture hash mismatch")
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    source = capture.get("source")
    metadata = capture.get("metadata")
    if (
        capture.get("schemaVersion") != 1
        or capture.get("operator") != OPERATOR
        or not isinstance(source, dict)
        or source.get("upstreamCommit") != UPSTREAM_COMMIT
        or source.get("bundleSha256") != BUNDLE_SHA256
        or not isinstance(metadata, dict)
        or metadata.get("headDim") != HEAD_DIM
        or metadata.get("heads") != 1
        or metadata.get("epsilon") != EPSILON_VALUE
        or metadata.get("ropeTable") != "sliding"
    ):
        raise SystemExit("DecodeQkNormRope capture identity or metadata mismatch")

    input_value = captured_tensor(capture, "k", [1, HEAD_DIM])
    weight = captured_tensor(capture, "kNormWeight", [HEAD_DIM])
    cosine = captured_tensor(capture, "ropeCos", [1, HALF_DIM])
    sine = captured_tensor(capture, "ropeSin", [1, HALF_DIM])
    expected_output = captured_tensor(capture, "kNormRopeOutput", [1, HEAD_DIM])
    computed_output = exact_norm_rope(input_value, weight, cosine, sine)
    expected_flat = expected_output.reshape(HEAD_DIM)
    changed = np.flatnonzero(computed_output != expected_flat)
    absolute_error = np.abs(computed_output - expected_flat)
    ulp_error = np.abs(ordered_f32(computed_output) - ordered_f32(expected_flat))
    maximum_absolute_error = float(np.max(absolute_error))
    maximum_ulp_error = int(np.max(ulp_error))
    if (
        maximum_absolute_error > float(MAX_CPU_ABSOLUTE_ERROR)
        or maximum_ulp_error > MAX_CPU_ULP_ERROR
    ):
        raise SystemExit(
            "Derived K norm/RoPE output exceeds the captured Metal arithmetic "
            f"bounds: absolute={maximum_absolute_error}, ulp={maximum_ulp_error}"
        )

    output_tensors = {
        "input": torch.from_numpy(input_value.reshape(HEAD_DIM).copy()),
        "weight": torch.from_numpy(weight.copy()),
        "cosine": torch.from_numpy(cosine.reshape(HALF_DIM).copy()),
        "sine": torch.from_numpy(sine.reshape(HALF_DIM).copy()),
        "output": torch.from_numpy(expected_output.reshape(HEAD_DIM).copy()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output_dir / "decode-k-norm-rope-layer0.safetensors"
    save_file(output_tensors, tensor_path)

    tokens = capture.get("tokens")
    if not isinstance(tokens, dict):
        raise SystemExit("Capture is missing token metadata")
    output_metadata = {
        "schemaVersion": 1,
        "sourceCapture": {
            "file": args.capture.name,
            "sha256": sha256_file(args.capture),
            "upstreamCommit": UPSTREAM_COMMIT,
            "bundleSha256": BUNDLE_SHA256,
        },
        "operator": OPERATOR,
        "decodeInputToken": tokens.get("decodeInputToken"),
        "position": tokens.get("position"),
        "headDim": HEAD_DIM,
        "heads": 1,
        "workgroupSize": WORKGROUP_SIZE,
        "epsilon": EPSILON_VALUE,
        "ropeTable": "sliding",
        "cpuReconstruction": {
            "differentValues": int(changed.size),
            "maximumAbsoluteError": maximum_absolute_error,
            "maximumUlpError": maximum_ulp_error,
            "note": "WGSL permits device-dependent inverseSqrt and contraction rounding; cancellation can amplify ULP distance near zero",
        },
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
    metadata_path = args.output_dir / "decode-k-norm-rope-layer0.json"
    metadata_path.write_text(
        json.dumps(output_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_metadata, indent=2))


if __name__ == "__main__":
    main()