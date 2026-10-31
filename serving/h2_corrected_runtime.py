"""Opt-in corrected H2 runtime, preserving the historical native benchmark code."""

from __future__ import annotations

import json
from pathlib import Path

from serving.backends.nanovllm import ROOT, digest
from serving.h2_numerical_policy import install_shape_policy
from serving.h2_runtime import check_capacity, close_engine, create_engine, load_inputs, provenance
from serving.h2_workloads import CORPUS, build_case, token_hash

DEFAULT_VALIDATION = ROOT / "config/serving/h2_corrected_validation.json"
CORRECTED_SOURCES = (
    "serving/h2_numerical_policy.py", "serving/h2_corrected_runtime.py",
    "serving/verify_h2_corrected.py", "scripts/run_h2_corrected_validation.py",
)


def load_corrected_inputs(path=DEFAULT_VALIDATION):
    validation = json.loads(path.read_text())
    pilot, config, h1, original = load_inputs(ROOT / validation["base_pilot"])
    if validation["forced_output_tokens"] != 64:
        raise ValueError("This validation contract fixes all 64 output steps")
    if validation["policy"] != dict(static_tables=True, paged_prefill=True,
            recompile_limit=64, fail_on_recompile_limit_hit=True):
        raise ValueError("Numerical policy differs from the qualified v1 controls")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / h1["model"]["path"], local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    pool = encode(CORPUS)
    cases = [original[name] for name in validation["original_case_ids"]]
    for dimensions in validation["additional_cases"]:
        case = build_case(pool, encode, **dimensions, block_size=config["block_size"],
            fraction=config["shared_prefix_fraction"])
        case["case_id"] += f"_seed{dimensions['seed']}"
        case["workload_sha256"] = token_hash({k: v for k, v in case.items() if k != "workload_sha256"})
        cases.append(case)
    for case in cases:
        check_capacity(pilot["engine"], case)
        if case["output_length"] != 64 or case["prompt_length"] < 64:
            raise ValueError("Reverse-prompt teacher forcing needs 64 input tokens")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Validation case IDs must be unique")
    return validation, pilot, config, h1, cases


def configure_compiler(validation):
    import torch._dynamo.config as dynamo_config
    dynamo_config.recompile_limit = validation["policy"]["recompile_limit"]
    dynamo_config.fail_on_recompile_limit_hit = validation["policy"]["fail_on_recompile_limit_hit"]
    dynamo_config.suppress_errors = False


def corrected_provenance(path, validation, pilot, config, h1):
    return {**provenance(ROOT / validation["base_pilot"], pilot, config, h1),
        "numerical_policy_id": validation["policy_id"], "numerical_policy": validation["policy"],
        "validation_config_sha256": digest(path),
        "corrected_source_sha256": {name: digest(ROOT / name) for name in CORRECTED_SOURCES}}


def create_corrected_engine(h1, pilot, switches, validation):
    configure_compiler(validation)
    llm, timings = create_engine(h1, pilot, switches)
    llm._h2_policy_restore = install_shape_policy(llm.model_runner,
        static_tables=validation["policy"]["static_tables"],
        paged_prefill=validation["policy"]["paged_prefill"])
    return llm, timings


def close_corrected_engine(llm):
    # Restore the bound methods and release their closures before upstream drops
    # the runner. Otherwise a retained restorer can keep model/Graph tensors live.
    llm._h2_policy_restore()
    del llm._h2_policy_restore
    close_engine(llm)
