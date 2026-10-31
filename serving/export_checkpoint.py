#!/usr/bin/env python3
"""Export a training checkpoint as a versioned RouteScale inference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from serving.artifact import export_inference_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("runs/checkpoints/g_serving/dense_baseline/model.pt"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/g_serving/g1_export/manifest.json"),
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument(
        "--inference-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_inference_artifact(
        checkpoint_path=args.checkpoint,
        artifact_path=args.artifact_path,
        manifest_path=args.manifest_path,
        tokenizer_name=args.tokenizer,
        inference_dtype=args.inference_dtype,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

