"""Reproduce the pinned upstream API on a small, explicitly non-formal workload."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import time
import traceback

from serving.backends.nanovllm import (
    DEFAULT_CONFIG, ROOT, base_result, digest, load_config, model_manifest,
    validate_environment, write_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=ROOT / "results/h_nanovllm/h1_smoke.json")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_environment(config)
    result = base_result(config, "functional_gpu_smoke")
    result["entrypoint_sha256"] = digest(Path(__file__))
    result["config_sha256"] = digest(args.config)
    phase = "model_manifest"
    llm = None
    try:
        result["model"] = model_manifest(config)
        import torch
        from transformers import AutoTokenizer
        from nanovllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("H1 generation requires CUDA")
        result["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_cuda": torch.version.cuda,
        }
        torch.manual_seed(config["seed"])
        path = str(ROOT / config["model"]["path"])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=config["enable_thinking"],
        ) for prompt in config["prompts"]]
        ids = [tokenizer.encode(prompt) for prompt in prompts]
        if any(len(tokens) + config["sampling"]["max_tokens"] > config["engine"]["max_model_len"] for tokens in ids):
            raise ValueError("Smoke requests exceed the configured model window")
        result["workload"] = {
            "prompts": config["prompts"], "prompt_token_ids": ids,
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "sampling": config["sampling"], "seed": config["seed"],
            "enable_thinking": config["enable_thinking"],
        }
        phase = "engine_initialization"
        torch.cuda.synchronize()
        started = time.perf_counter()
        llm = LLM(path, **config["engine"])
        torch.cuda.synchronize()
        result["initialization_wall_seconds"] = time.perf_counter() - started
        result["engine"] = {**config["engine"], "num_kvcache_blocks": len(llm.scheduler.block_manager.blocks)}
        phase = "generation"
        started = time.perf_counter()
        outputs = llm.generate(ids, SamplingParams(**config["sampling"]), use_tqdm=False)
        torch.cuda.synchronize()
        result["generation_wall_seconds_diagnostic_only"] = time.perf_counter() - started
        result["outputs"] = outputs
        result["output_token_ids_sha256"] = hashlib.sha256(json.dumps(
            [out["token_ids"] for out in outputs]
        ).encode()).hexdigest()
        result["requests_completed"] = len(outputs)
        result["remaining_running_requests"] = len(llm.scheduler.running)
        result["remaining_waiting_requests"] = len(llm.scheduler.waiting)
        result["used_blocks_after_completion"] = len(llm.scheduler.block_manager.used_block_ids)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        assert len(outputs) == len(ids) and all(out["token_ids"] for out in outputs)
        assert all(len(out["token_ids"]) <= config["sampling"]["max_tokens"] for out in outputs)
        assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
        result["status"] = "pass"
    except Exception:
        result.update(status="fail", failure_phase=phase, traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                atexit.unregister(llm.exit)
                llm.exit()
        finally:
            write_result(result, args.result)
    print(f"H1 generation: {result['status']}, {result['requests_completed']} requests -> {args.result}")


if __name__ == "__main__":
    main()
