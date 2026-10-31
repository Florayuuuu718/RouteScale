"""Single-process timing primitives for the G3 native serving baseline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from model import GPT


BACKENDS = {"naive", "kv_cache"}
PREFIX_MODES = {"none", "high"}


@dataclass
class GenerationTiming:
    token_ids: torch.Tensor
    predicted_token_ids: torch.Tensor
    ttft_ms: float
    decode_step_ms: list[float]
    e2e_device_ms: float
    e2e_wall_ms: float
    cache_rebuilds: int
    cache_allocated_bytes: int


def build_prompt_batch(
    data_path: str | Path,
    *,
    batch_size: int,
    prompt_length: int,
    prefix_mode: str,
    seed: int,
    shared_prefix_fraction: float = 0.75,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build deterministic exact-length prompts from a tokenized validation set."""

    if batch_size <= 0 or prompt_length <= 0:
        raise ValueError("batch_size and prompt_length must be positive")
    if prefix_mode not in PREFIX_MODES:
        raise ValueError(f"unsupported prefix mode: {prefix_mode}")
    if not 0.0 <= shared_prefix_fraction <= 1.0:
        raise ValueError("shared_prefix_fraction must be between zero and one")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(data) <= prompt_length:
        raise ValueError("token file is too short for requested prompt length")
    generator = np.random.Generator(np.random.PCG64(seed))
    max_start = len(data) - prompt_length
    starts = generator.integers(0, max_start + 1, size=batch_size, dtype=np.int64)
    prompts = np.empty((batch_size, prompt_length), dtype=np.int64)
    if prefix_mode == "none":
        for index, start in enumerate(starts):
            prompts[index] = data[start : start + prompt_length]
        common_tokens = 0
    else:
        common_tokens = int(prompt_length * shared_prefix_fraction)
        shared_start = int(generator.integers(0, max_start + 1))
        shared = np.asarray(
            data[shared_start : shared_start + common_tokens], dtype=np.int64
        )
        suffix_length = prompt_length - common_tokens
        for index, start in enumerate(starts):
            prompts[index, :common_tokens] = shared
            if suffix_length:
                prompts[index, common_tokens:] = data[
                    start : start + suffix_length
                ]
    canonical = np.asarray(prompts, dtype="<i8")
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return torch.from_numpy(prompts), {
        "token_file": Path(data_path).as_posix(),
        "seed": seed,
        "prefix_mode": prefix_mode,
        "requested_shared_prefix_fraction": shared_prefix_fraction,
        "shared_prefix_tokens": common_tokens,
        "prompt_token_ids_sha256": digest,
    }


def build_decode_batch(
    data_path: str | Path,
    *,
    batch_size: int,
    output_length: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build fixed continuation inputs so backend timing follows one token path."""

    if batch_size <= 0 or output_length <= 0:
        raise ValueError("batch_size and output_length must be positive")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(data) <= output_length:
        raise ValueError("token file is too short for requested output length")
    generator = np.random.Generator(np.random.PCG64(seed))
    max_start = len(data) - output_length
    starts = generator.integers(0, max_start + 1, size=batch_size, dtype=np.int64)
    tokens = np.stack(
        [
            np.asarray(data[start : start + output_length], dtype=np.int64)
            for start in starts
        ]
    )
    canonical = np.asarray(tokens, dtype="<i8")
    return torch.from_numpy(tokens), {
        "decode_input_policy": "fixed_validation_tokens",
        "decode_seed": seed,
        "decode_token_ids_sha256": hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest(),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cuda_timed_call(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.cuda.Event, torch.cuda.Event],
]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    generated, token, predicted_token = operation()
    end.record()
    return generated, token, predicted_token, (start, end)


def _cpu_timed_call(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    started = time.perf_counter()
    generated, token, predicted_token = operation()
    return (
        generated,
        token,
        predicted_token,
        (time.perf_counter() - started) * 1000.0,
    )


@torch.inference_mode()
def generate_timed(
    model: GPT,
    prompt_ids: torch.Tensor,
    *,
    output_length: int,
    backend: str,
    forced_token_ids: torch.Tensor | None = None,
) -> GenerationTiming:
    """Generate one synchronous batch and time TTFT plus every later token."""

    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [batch, positive sequence]")
    if forced_token_ids is not None and forced_token_ids.shape != (
        prompt_ids.shape[0],
        output_length,
    ):
        raise ValueError("forced_token_ids must have shape [batch, output_length]")
    parameter = next(model.parameters())
    if prompt_ids.device != parameter.device:
        raise ValueError("prompt and model must be on the same device")
    if forced_token_ids is not None and forced_token_ids.device != parameter.device:
        raise ValueError("forced tokens and model must be on the same device")
    is_cuda = parameter.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(parameter.device)
    wall_started = time.perf_counter()

    generated = prompt_ids
    cache = None
    cache_rebuilds = 0
    cache_allocated_bytes = 0
    timings: list[Any] = []
    next_token = None
    predicted_tokens: list[torch.Tensor] = []

    for step in range(output_length):
        forced_token = (
            None
            if forced_token_ids is None
            else forced_token_ids[:, step : step + 1]
        )
        if backend == "naive":
            def operation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                context = generated[:, -model.config.block_size :]
                logits, _ = model(context)
                predicted = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token = predicted if forced_token is None else forced_token
                return torch.cat((generated, token), dim=1), token, predicted
        else:
            if step == 0:
                cache = model.allocate_kv_cache(prompt_ids.shape[0])
                context = generated[:, -model.config.block_size :]
            elif cache.length == cache.capacity:
                cache = model.allocate_kv_cache(prompt_ids.shape[0])
                context = generated[:, -model.config.block_size :]
                cache_rebuilds += 1
            else:
                context = next_token
            cache_allocated_bytes = cache.allocated_bytes

            def operation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                logits = model.forward_with_kv_cache(context, cache)[:, -1, :]
                predicted = torch.argmax(logits, dim=-1, keepdim=True)
                token = predicted if forced_token is None else forced_token
                return torch.cat((generated, token), dim=1), token, predicted

        if is_cuda:
            generated, next_token, predicted_token, timing = _cuda_timed_call(operation)
        else:
            generated, next_token, predicted_token, timing = _cpu_timed_call(operation)
        timings.append(timing)
        predicted_tokens.append(predicted_token)

    if is_cuda:
        torch.cuda.synchronize(parameter.device)
        step_ms = [start.elapsed_time(end) for start, end in timings]
    else:
        step_ms = timings
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    return GenerationTiming(
        token_ids=generated,
        predicted_token_ids=torch.cat(predicted_tokens, dim=1),
        ttft_ms=step_ms[0],
        decode_step_ms=step_ms[1:],
        e2e_device_ms=sum(step_ms),
        e2e_wall_ms=wall_ms,
        cache_rebuilds=cache_rebuilds,
        cache_allocated_bytes=cache_allocated_bytes,
    )


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing list")
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize_measurements(
    measurements: list[GenerationTiming],
    *,
    batch_size: int,
    prompt_length: int,
    output_length: int,
) -> dict[str, Any]:
    if not measurements:
        raise ValueError("measurements must not be empty")
    ttft = [item.ttft_ms for item in measurements]
    e2e_device = [item.e2e_device_ms for item in measurements]
    e2e_wall = [item.e2e_wall_ms for item in measurements]
    decode = [value for item in measurements for value in item.decode_step_ms]
    total_wall_seconds = sum(e2e_wall) / 1000.0
    total_prefill_seconds = sum(ttft) / 1000.0
    total_decode_seconds = sum(decode) / 1000.0
    request_count = batch_size * len(measurements)
    output_tokens = request_count * output_length
    decode_tokens = request_count * max(output_length - 1, 0)
    return {
        "successful_requests": request_count,
        "input_tokens": request_count * prompt_length,
        "output_tokens": output_tokens,
        "ttft_ms": ttft,
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "ttft_p99_ms": percentile(ttft, 99),
        "decode_step_ms": decode,
        "tpot_p50_ms": percentile(decode, 50) if decode else None,
        "tpot_p95_ms": percentile(decode, 95) if decode else None,
        "tpot_p99_ms": percentile(decode, 99) if decode else None,
        "e2e_device_ms": e2e_device,
        "e2e_wall_ms": e2e_wall,
        "e2e_wall_p50_ms": percentile(e2e_wall, 50),
        "e2e_wall_p95_ms": percentile(e2e_wall, 95),
        "e2e_wall_p99_ms": percentile(e2e_wall, 99),
        "request_throughput_per_second": request_count / total_wall_seconds,
        "output_tokens_per_second": output_tokens / total_wall_seconds,
        "prefill_input_tokens_per_second": (
            request_count * prompt_length / total_prefill_seconds
        ),
        "decode_tokens_per_second": (
            decode_tokens / total_decode_seconds
            if decode_tokens and total_decode_seconds > 0
            else None
        ),
        "cache_rebuilds": [item.cache_rebuilds for item in measurements],
        "cache_allocated_bytes": max(
            item.cache_allocated_bytes for item in measurements
        ),
        "output_token_ids_sha256": [
            tensor_sha256(item.token_ids[:, -output_length:])
            for item in measurements
        ],
        "output_token_ids_sample": measurements[0].token_ids[
            0, -output_length:
        ].tolist(),
        "outputs_deterministic": len(
            {
                tensor_sha256(item.token_ids[:, -output_length:])
                for item in measurements
            }
        ) == 1,
        "prediction_token_ids_sha256": [
            tensor_sha256(item.predicted_token_ids) for item in measurements
        ],
        "prediction_token_ids_sample": measurements[0].predicted_token_ids[
            0
        ].tolist(),
        "predictions_deterministic": len(
            {tensor_sha256(item.predicted_token_ids) for item in measurements}
        ) == 1,
    }

