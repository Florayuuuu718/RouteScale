"""Prepare a fixed TinyStories subset or the complete dataset for nanoGPT."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "roneneldan/TinyStories"
DEFAULT_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER_NAME = "gpt2"
DEBUG_LIMITS = {"train": 10_000, "validation": 1_000}
WRITE_BUFFER_TOKENS = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("debug", "full"), default="debug")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    split: str,
    output_path: Path,
    revision: str,
    story_limit: int | None,
    encoder: tiktoken.Encoding,
) -> dict[str, int | str | None]:
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        revision=revision,
        streaming=True,
    )
    if story_limit is not None:
        dataset = dataset.take(story_limit)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    story_count = 0
    token_count = 0
    max_token_id = -1
    token_buffer: list[int] = []

    source_iterator = iter(dataset)
    progress = tqdm(source_iterator, total=story_limit, unit="stories", desc=split)
    try:
        with temporary_path.open("wb") as output:
            for example in progress:
                ids = encoder.encode_ordinary(example["text"])
                ids.append(encoder.eot_token)
                token_buffer.extend(ids)
                story_count += 1
                token_count += len(ids)
                max_token_id = max(max_token_id, max(ids))

                if len(token_buffer) >= WRITE_BUFFER_TOKENS:
                    np.asarray(token_buffer, dtype=np.uint16).tofile(output)
                    token_buffer.clear()

            if token_buffer:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output)
    finally:
        progress.close()
        close_iterator = getattr(source_iterator, "close", None)
        if close_iterator is not None:
            close_iterator()
        del progress, source_iterator, dataset
        gc.collect()

    if story_count == 0 or token_count == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"{split} produced no data")
    if max_token_id >= 2**16:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"token id {max_token_id} does not fit into uint16")

    temporary_path.replace(output_path)
    return {
        "source_split": split,
        "story_limit": story_limit,
        "story_count": story_count,
        "token_count": token_count,
        "max_token_id": max_token_id,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or repo_root / "data" / f"tinystories_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    limits = DEBUG_LIMITS if args.mode == "debug" else {"train": None, "validation": None}
    results = {
        "train": write_split(
            "train", output_dir / "train.bin", args.revision, limits["train"], encoder
        ),
        "validation": write_split(
            "validation", output_dir / "val.bin", args.revision, limits["validation"], encoder
        ),
    }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "dataset_id": DATASET_ID,
        "dataset_revision": args.revision,
        "selection": "first N rows in source order" if args.mode == "debug" else "complete splits",
        "tokenizer": TOKENIZER_NAME,
        "software_versions": {
            "datasets": importlib.metadata.version("datasets"),
            "tiktoken": importlib.metadata.version("tiktoken"),
        },
        "vocab_size": encoder.n_vocab,
        "separator": "GPT-2 end-of-text token appended to every story",
        "separator_token_id": encoder.eot_token,
        "dtype": "uint16",
        "splits": results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": encoder.n_vocab}, handle)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
