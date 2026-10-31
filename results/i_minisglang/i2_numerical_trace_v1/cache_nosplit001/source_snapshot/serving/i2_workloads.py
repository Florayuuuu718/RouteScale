"""Frozen raw-token cohorts for the controlled I2 scheduler replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/serving/i2_single_gpu.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_workload(config, tokenizer):
    from serving.h2_workloads import CORPUS
    pool = tokenizer.encode(CORPUS, add_special_tokens=False)
    unique = sorted(set(pool) - set(tokenizer.all_special_ids))
    rng = random.Random(config["seed"])
    rng.shuffle(unique)
    cases = []
    for batch in config["batch_sizes"]:
        for prefix in ("zero", "shared"):
            shared = [pool[i % len(pool)] for i in range(config["shared_prefix_tokens"])] if prefix == "shared" else []
            case = dict(case_id=f"b{batch}_{prefix}", batch_size=batch, prefix_mode=prefix,
                        prompt_tokens=config["prompt_tokens"], output_tokens=config["output_tokens"], waves={})
            for phase_index, phase in enumerate(("cold", "warm")):
                wave = []
                for slot in range(batch):
                    marker = unique[phase_index * batch + slot]
                    offset = rng.randrange(len(pool))
                    length = config["prompt_tokens"] - len(shared) - 1
                    ids = shared + [marker] + [pool[(offset + j) % len(pool)] for j in range(length)]
                    wave.append(dict(request_id=f"{case['case_id']}_{phase}_{slot}", input_ids=ids,
                        continuation_ids=list(reversed(ids[-config["output_tokens"]:])),
                        input_sha256=token_hash(ids)))
                case["waves"][phase] = wave
            case["sha256"] = token_hash(case)
            cases.append(case)
    result = dict(stage="I2", config_sha256=digest(CONFIG), builder_sha256=digest(__file__),
        corpus_sha256=hashlib.sha256(CORPUS.encode()).hexdigest(), cases=cases)
    validate_workload(result, config)
    return result


def validate_workload(workload, config):
    expected = {f"b{batch}_{prefix}" for batch in config["batch_sizes"] for prefix in ("zero", "shared")}
    if {c["case_id"] for c in workload["cases"]} != expected:
        raise ValueError("Workload case coverage changed")
    all_ids = []
    for case in workload["cases"]:
        if case["sha256"] != token_hash({k: v for k, v in case.items() if k != "sha256"}):
            raise ValueError("Workload hash mismatch")
        rows = [r for phase in ("cold", "warm") for r in case["waves"][phase]]
        prompts = [r["input_ids"] for r in rows]
        common = 0
        for column in zip(*prompts):
            if len(set(column)) != 1:
                break
            common += 1
        if common != (config["shared_prefix_tokens"] if case["prefix_mode"] == "shared" else 0):
            raise ValueError("Exact shared-prefix boundary changed")
        for r in rows:
            all_ids.append(r["request_id"])
            if (len(r["input_ids"]) != config["prompt_tokens"] or
                len(r["continuation_ids"]) != config["output_tokens"] or
                r["input_sha256"] != token_hash(r["input_ids"])):
                raise ValueError("Request length or input hash changed")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate workload IDs")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    config = json.loads(CONFIG.read_text())
    tok = AutoTokenizer.from_pretrained(ROOT / config["model_path"], local_files_only=True)
    result = build_workload(config, tok)
    path = ROOT / config["workload_path"]
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("Preserve frozen workload; choose a new config/path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(path=str(path), sha256=digest(path), cases=len(result["cases"]))))
