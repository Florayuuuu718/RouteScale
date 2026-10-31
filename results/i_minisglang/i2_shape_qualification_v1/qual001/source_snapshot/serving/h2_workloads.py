"""Build fixed Qwen token workloads; no model weights or GPU are needed."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path
import random

from serving.backends.nanovllm import ROOT, digest, load_config, write_result


CORPUS = (
    "A library keeps a record of every book that readers borrow. Each morning the librarian "
    "checks the returns, repairs damaged pages, and places new books on a wooden shelf. "
    "A small research team compares computer systems using the same inputs. They record "
    "the time, memory use, and results before deciding which change helps their workload. "
    "In the garden a child plants seeds beside a stone path. Rain fills a blue bucket, "
    "and birds wait in the branches until the child walks back to the house. "
    "A train crosses the valley every afternoon. The station clock helps passengers plan "
    "their journeys, while engineers inspect the tracks and prepare the next departure. "
    "When a request arrives at a server, it joins a queue. A scheduler chooses work, "
    "allocates memory, and returns resources after the request has finished. "
)


def token_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def common_prefix_length(prompts: list[list[int]]) -> int:
    for index, column in enumerate(zip(*prompts)):
        if len(set(column)) != 1:
            return index
    return min(map(len, prompts))


def build_case(pool: list[int], encode_marker, *, batch: int, prompt: int, output: int,
               prefix_mode: str, seed: int, block_size: int = 256, fraction: float = 0.75) -> dict:
    if not pool or min(batch, prompt, output, block_size) <= 0:
        raise ValueError("Nonempty token pool and positive dimensions required")
    if prefix_mode not in ("none", "high") or not 0 <= fraction < 1:
        raise ValueError("Invalid prefix workload")
    rng = random.Random(seed)
    def take(length):
        offset = rng.randrange(len(pool))
        return [pool[(offset + i) % len(pool)] for i in range(length)]
    shared = take(int(prompt * fraction)) if prefix_mode == "high" else []
    waves = {}
    for phase in ("cold", "warm"):
        requests = []
        for slot in range(batch):
            marker = encode_marker(f"Record {int(phase == 'warm') * batch + slot}: ")
            if len(shared) + len(marker) >= prompt:
                raise ValueError("Request marker does not fit after the shared prefix")
            tokens = shared + marker + take(prompt - len(shared) - len(marker))
            requests.append({"request_id": f"{phase}_{slot}", "prompt_token_ids": tokens,
                             "max_tokens": output})
        waves[phase] = requests
    prompts = [r["prompt_token_ids"] for phase in waves.values() for r in phase]
    if len({tuple(p) for p in prompts}) != 2 * batch:
        raise ValueError("Cold and warm waves must contain distinct requests")
    if prefix_mode == "none" and len({tuple(p[:block_size]) for p in prompts}) != len(prompts):
        raise ValueError("No-prefix requests unexpectedly share a first block")
    shared_length = common_prefix_length(prompts)
    reusable = min(shared_length // block_size, (prompt - 1) // block_size) * block_size
    result = {"case_id": f"b{batch}_p{prompt}_o{output}_{prefix_mode}",
        "batch_size": batch, "prompt_length": prompt, "output_length": output,
        "prefix_mode": prefix_mode, "requested_shared_prefix_tokens": len(shared),
        "actual_shared_prefix_tokens": shared_length,
        "expected_warm_cached_tokens_per_request": reusable, "waves": waves}
    result["prompt_stream_sha256"] = token_hash(prompts)
    result["workload_sha256"] = token_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/serving/h2_nanovllm.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    h1 = load_config(ROOT / config["h1_config"])
    from transformers import AutoTokenizer
    directory = ROOT / h1["model"]["path"]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        metadata = directory / ".cache/huggingface/download" / f"{name}.metadata"
        if metadata.read_text().splitlines()[0] != h1["model"]["revision"]:
            raise ValueError(f"Tokenizer revision mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    pool = encode(CORPUS)
    cases = []
    for batch, prompt, output, prefix in product(config["batch_sizes"], config["prompt_lengths"],
                                                config["output_lengths"], config["prefix_modes"]):
        if prompt + output > config["max_model_len"]:
            raise ValueError("Workload exceeds model window")
        case = build_case(pool, encode, batch=batch, prompt=prompt, output=output,
            prefix_mode=prefix, seed=config["seed"], block_size=config["block_size"],
            fraction=config["shared_prefix_fraction"])
        if any(set(r["prompt_token_ids"]) & set(tokenizer.all_special_ids)
               for wave in case["waves"].values() for r in wave):
            raise ValueError("Prompt includes a special token")
        cases.append(case)
    payload = {"schema_version": 1, "stage": "H2", "model": h1["model"],
        "tokenizer_implementation": {package: version(package) for package in ("transformers", "tokenizers")},
        "tokenizer_sha256": {name: digest(directory / name) for name in
            ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")},
        "config_sha256": digest(args.config), "builder_sha256": digest(Path(__file__)),
        "corpus_sha256": hashlib.sha256(CORPUS.encode()).hexdigest(), "cases": cases}
    output = ROOT / config["workload_path"]
    write_result(payload, output)
    manifest = {**{key: value for key, value in payload.items() if key != "cases"},
        "status": "pass", "evidence_class": "cpu_workload_preparation",
        "workload_path": config["workload_path"], "workload_file_sha256": digest(output),
        "case_count": len(cases), "cases": [
            {key: value for key, value in case.items() if key != "waves"} for case in cases]}
    write_result(manifest, ROOT / "results/h_nanovllm/h2_workload_manifest.json")
    selected = {case["case_id"]: case for case in cases if case["case_id"] in config["pilot_case_ids"]}
    if set(selected) != set(config["pilot_case_ids"]):
        raise ValueError("Pilot refers to a missing workload case")
    prerequisites = {name: json.loads((ROOT / "results/h_nanovllm" / name).read_text()).get("status")
        if (ROOT / "results/h_nanovllm" / name).exists() else "missing"
        for name in config["required_h1_results"]}
    plan = {"schema_version": 1, "stage": "H2", "evidence_class": "cpu_execution_plan",
        "gpu_execution_allowed": False,
        "h1_prerequisites": prerequisites,
        "pending_requirements": (["H1 acceptance"] if set(prerequisites.values()) != {"pass"} else [])
            + ["common GPU KV-pool preflight", "GPU timing entry point"],
        "planned_processes": [{"case_id": case_id, "workload_sha256": selected[case_id]["workload_sha256"],
            "variant": name, "switches": switches, "run_id": f"run{index}"}
            for case_id in config["pilot_case_ids"] for name, switches in config["variants"].items()
            for index in range(1, config["independent_runs"] + 1)]}
    write_result(plan, ROOT / "results/h_nanovllm/h2_execution_plan.json")
    print(f"H2 fixed workloads: {len(cases)} cases -> {output}")


if __name__ == "__main__":
    main()
