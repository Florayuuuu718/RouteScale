"""Versioned inference artifacts for RouteScale GPT checkpoints."""

from __future__ import annotations

from dataclasses import asdict, fields
import importlib.metadata
from pathlib import Path
from typing import Any

import torch

from distributed_common import (
    atomic_json_dump,
    atomic_torch_save,
    sha256_file,
    state_dict_sha256,
)
from model import GPT, GPTConfig


ARTIFACT_FORMAT = "routescale.inference"
ARTIFACT_VERSION = 1
COMPILE_PREFIX = "_orig_mod."
SUPPORTED_DTYPES = {"float32", "bfloat16", "float16"}


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the wrapper prefix produced by ``torch.compile`` checkpoints."""

    normalized: dict[str, Any] = {}
    for original_name, value in state_dict.items():
        name = (
            original_name[len(COMPILE_PREFIX) :]
            if original_name.startswith(COMPILE_PREFIX)
            else original_name
        )
        if name in normalized:
            raise ValueError(f"state_dict key collision after prefix cleanup: {name}")
        normalized[name] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return normalized


def _required_tensor(
    state_dict: dict[str, Any], name: str
) -> torch.Tensor:
    value = state_dict.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint is missing tensor {name!r}")
    return value


def model_config_from_checkpoint(
    checkpoint: dict[str, Any], state_dict: dict[str, Any]
) -> GPTConfig:
    """Recover a dense ``GPTConfig`` from nanoGPT or RouteScale C checkpoints."""

    config_fields = {field.name for field in fields(GPTConfig)}
    source: dict[str, Any] = {}
    if isinstance(checkpoint.get("model_args"), dict):
        source.update(checkpoint["model_args"])
    if isinstance(checkpoint.get("signature"), dict):
        for name in config_fields:
            if name in checkpoint["signature"]:
                source.setdefault(name, checkpoint["signature"][name])

    token_embedding = _required_tensor(state_dict, "transformer.wte.weight")
    position_embedding = _required_tensor(state_dict, "transformer.wpe.weight")
    source.setdefault("vocab_size", int(token_embedding.shape[0]))
    source.setdefault("n_embd", int(token_embedding.shape[1]))
    source.setdefault("block_size", int(position_embedding.shape[0]))

    if "n_layer" not in source:
        layer_indices = {
            int(name.split(".")[2])
            for name in state_dict
            if name.startswith("transformer.h.") and name.split(".")[2].isdigit()
        }
        if not layer_indices:
            raise ValueError("cannot infer n_layer from checkpoint")
        source["n_layer"] = max(layer_indices) + 1
    if "n_head" not in source:
        raise ValueError("checkpoint does not record n_head")

    config = GPTConfig(**{name: source[name] for name in config_fields if name in source})
    if config.moe_num_experts != 0:
        raise ValueError("G1 accepts a dense checkpoint only")
    if token_embedding.shape != (config.vocab_size, config.n_embd):
        raise ValueError("token embedding shape does not match recovered GPTConfig")
    if position_embedding.shape != (config.block_size, config.n_embd):
        raise ValueError("position embedding shape does not match recovered GPTConfig")
    return config


def tokenizer_metadata(name: str) -> dict[str, Any]:
    # Export must remain offline and deterministic. The GPT-2 encoding contract
    # is fixed; loading its merge table is only necessary when text is encoded.
    if name != "gpt2":
        raise ValueError(f"unsupported offline tokenizer metadata: {name}")
    return {
        "implementation": "tiktoken",
        "implementation_version": importlib.metadata.version("tiktoken"),
        "name": name,
        "vocab_size": 50257,
        "special_tokens": {"end_of_text": 50256},
        "story_separator": "end_of_text",
    }


def _checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("training checkpoint does not contain a model state_dict")
    return normalize_state_dict(state)


def load_training_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GPT, GPTConfig, dict[str, Any]]:
    """Strictly load a training checkpoint without restoring training-only state."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must be a dictionary")
    state_dict = _checkpoint_state(checkpoint)
    config = model_config_from_checkpoint(checkpoint, state_dict)
    model = GPT(config)
    model.load_state_dict(state_dict, strict=True)
    if model.transformer.wte.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("GPT weight tying was not preserved after strict load")
    model.eval().to(device=device, dtype=dtype)
    return model, config, checkpoint


def export_inference_artifact(
    *,
    checkpoint_path: str | Path,
    artifact_path: str | Path,
    manifest_path: str | Path,
    tokenizer_name: str = "gpt2",
    inference_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Export only inference state and write a lightweight provenance manifest."""

    if inference_dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported inference dtype: {inference_dtype}")
    source_path = Path(checkpoint_path)
    destination = Path(artifact_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must be a dictionary")
    state_dict = _checkpoint_state(checkpoint)
    config = model_config_from_checkpoint(checkpoint, state_dict)
    tokenizer = tokenizer_metadata(tokenizer_name)
    if tokenizer["special_tokens"]["end_of_text"] >= config.vocab_size:
        raise ValueError("tokenizer special token does not fit model vocabulary")

    token_embedding = _required_tensor(state_dict, "transformer.wte.weight")
    lm_head = _required_tensor(state_dict, "lm_head.weight")
    if not torch.equal(token_embedding, lm_head):
        raise ValueError("checkpoint token embedding and LM head are not tied-equivalent")

    # Strict load during export catches missing, unexpected and shape-mismatched keys.
    validation_model = GPT(config)
    validation_model.load_state_dict(state_dict, strict=True)
    tied_storage = (
        validation_model.transformer.wte.weight.data_ptr()
        == validation_model.lm_head.weight.data_ptr()
    )
    if not tied_storage:
        raise RuntimeError("model construction did not preserve tied embedding storage")
    del validation_model

    state_hash = state_dict_sha256(state_dict)
    weight_dtypes = sorted(
        {str(value.dtype).removeprefix("torch.") for value in state_dict.values()
         if isinstance(value, torch.Tensor)}
    )
    source_training_dtype = None
    if isinstance(checkpoint.get("config"), dict):
        source_training_dtype = checkpoint["config"].get("dtype")
    if source_training_dtype is None and isinstance(checkpoint.get("signature"), dict):
        source_training_dtype = checkpoint["signature"].get("dtype")

    provenance = {
        "source_checkpoint": source_path.as_posix(),
        "source_checkpoint_bytes": source_path.stat().st_size,
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_format": (
            "nanogpt" if "model_args" in checkpoint else "routescale_distributed"
        ),
        "training_step": checkpoint.get("iter_num", checkpoint.get("next_update")),
        "best_validation_loss": (
            float(checkpoint["best_val_loss"])
            if checkpoint.get("best_val_loss") is not None
            else None
        ),
        "source_training_dtype": source_training_dtype,
    }
    metadata = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "model_config": asdict(config),
        "tokenizer": tokenizer,
        "inference_dtype": inference_dtype,
        "stored_weight_dtypes": weight_dtypes,
        "state_dict_sha256": state_hash,
        "tied_embedding_verified": True,
        "provenance": provenance,
    }
    artifact = {**metadata, "state_dict": state_dict}
    atomic_torch_save(artifact, destination)
    manifest = {
        **metadata,
        "artifact_path": destination.as_posix(),
        "artifact_bytes": destination.stat().st_size,
        "artifact_sha256": sha256_file(destination),
        "excluded_training_state": ["optimizer", "rng", "data_window_progress"],
    }
    atomic_json_dump(manifest, manifest_path)
    return manifest


def load_inference_artifact(
    artifact_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[GPT, dict[str, Any]]:
    """Validate and strictly load a RouteScale inference artifact."""

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError("inference artifact must be a dictionary")
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise ValueError("unknown inference artifact format")
    if artifact.get("format_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported inference artifact version")
    state_dict = artifact.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("inference artifact is missing state_dict")
    if state_dict_sha256(state_dict) != artifact.get("state_dict_sha256"):
        raise ValueError("inference artifact state_dict hash mismatch")

    config = GPTConfig(**artifact["model_config"])
    model = GPT(config)
    model.load_state_dict(state_dict, strict=True)
    if model.transformer.wte.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("GPT weight tying was not preserved after artifact load")
    model.eval().to(device=device, dtype=dtype)
    metadata = {name: value for name, value in artifact.items() if name != "state_dict"}
    return model, metadata

