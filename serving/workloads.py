"""Fixed, hashed serving correctness workloads."""

from __future__ import annotations

import hashlib

import torch
import tiktoken


G1_PROMPTS = {
    "short": "Once upon a time",
    "medium": (
        "Once upon a time, a little fox lived beside a quiet forest. "
        "Every morning it carried berries to its friends and listened to the birds."
    ),
    "long": (
        "Once upon a time, a curious child found a small wooden door beneath an old tree. "
        "The door opened into a garden filled with silver flowers, warm lanterns, and animals "
        "that could speak. The child promised to keep the garden safe, returned every afternoon, "
        "and slowly learned why kindness mattered more than treasure. "
    ) * 2,
}


def encode_prompts(prompts: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
    encoder = tiktoken.get_encoding("gpt2")
    selected = G1_PROMPTS if prompts is None else prompts
    return {
        name: torch.tensor(encoder.encode(text), dtype=torch.long).unsqueeze(0)
        for name, text in selected.items()
    }


def prompt_set_sha256(prompts: dict[str, str] | None = None) -> str:
    selected = G1_PROMPTS if prompts is None else prompts
    serialized = "\n".join(f"{name}:{selected[name]}" for name in sorted(selected))
    return hashlib.sha256(serialized.encode()).hexdigest()

