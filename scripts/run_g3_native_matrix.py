#!/usr/bin/env python3
"""Run the G3 single-GPU matrix as isolated, sequential child processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import atomic_json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/serving/g3_native.json")
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/g_serving/g3_native")
    )
    parser.add_argument("--gpu", default="0", help="one CUDA device index only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "artifact",
        "data_path",
        "dtype",
        "decode_input_policy",
        "backends",
        "prefix_modes",
        "batch_sizes",
        "prompt_lengths",
        "output_lengths",
        "warmup_repeats",
        "measure_repeats",
        "independent_runs",
        "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"G3 config is missing fields: {missing}")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported G3 config schema")
    return config


def _jobs(config: dict[str, Any], results_dir: Path):
    matrix = itertools.product(
        config["backends"],
        config["prefix_modes"],
        config["batch_sizes"],
        config["prompt_lengths"],
        config["output_lengths"],
        range(1, config["independent_runs"] + 1),
    )
    for backend, prefix, batch, prompt, output, run_index in matrix:
        run_id = f"run{run_index}"
        output_path = (
            results_dir
            / "raw"
            / backend
            / prefix
            / f"b{batch}_p{prompt}_o{output}"
            / f"{run_id}.json"
        )
        yield {
            "backend": backend,
            "prefix": prefix,
            "batch": batch,
            "prompt": prompt,
            "output": output,
            "run_id": run_id,
            "output_path": output_path,
        }


def main() -> None:
    args = parse_args()
    if "," in args.gpu:
        raise ValueError("G3 accepts exactly one GPU index")
    config = _load_config(args.config)
    for name in ("artifact", "data_path"):
        if not (REPOSITORY_ROOT / config[name]).is_file():
            raise FileNotFoundError(f"G3 {name} does not exist: {config[name]}")
    jobs = list(_jobs(config, args.results_dir))
    print(f"G3 matrix contains {len(jobs)} isolated processes")
    if args.dry_run:
        for job in jobs[:5]:
            print(job)
        if len(jobs) > 5:
            print(f"... {len(jobs) - 5} more")
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    for index, job in enumerate(jobs, start=1):
        output_path = job["output_path"]
        if args.resume and output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "pass" and existing.get("evidence_class") == "formal_gpu":
                print(f"[{index}/{len(jobs)}] skip {output_path}")
                continue
        command = [
            sys.executable,
            "-m",
            "serving.benchmark_native",
            f"--artifact={config['artifact']}",
            f"--data-path={config['data_path']}",
            f"--backend={job['backend']}",
            f"--prefix-mode={job['prefix']}",
            f"--batch-size={job['batch']}",
            f"--prompt-length={job['prompt']}",
            f"--output-length={job['output']}",
            f"--dtype={config['dtype']}",
            f"--decode-input-policy={config['decode_input_policy']}",
            "--device=cuda",
            f"--warmup-repeats={config['warmup_repeats']}",
            f"--measure-repeats={config['measure_repeats']}",
            f"--seed={config['seed']}",
            f"--run-id={job['run_id']}",
            f"--result-path={output_path}",
        ]
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        completed = subprocess.run(
            command, cwd=REPOSITORY_ROOT, env=environment, check=False
        )
        if completed.returncode != 0:
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "stage": "G3",
                    "benchmark": "native_serving",
                    "status": "fail",
                    "evidence_class": "formal_gpu_failure",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": job["run_id"],
                    "configuration": {
                        **job,
                        "output_path": output_path.as_posix(),
                    },
                    "failure": {
                        "phase": "isolated_child_process",
                        "returncode": completed.returncode,
                        "command": command,
                    },
                },
                output_path,
            )
            raise RuntimeError(f"G3 child failed; record written to {output_path}")

    summary_path = args.results_dir / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_g3_serving.py",
            f"--raw-dir={args.results_dir / 'raw'}",
            f"--output={summary_path}",
            f"--expected-runs={config['independent_runs']}",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

