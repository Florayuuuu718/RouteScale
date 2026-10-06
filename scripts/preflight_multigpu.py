#!/usr/bin/env python3
"""Record RouteScale host/GPU/software/data readiness before paid GPU runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
import torch.distributed as dist

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed_common import (  # noqa: E402
    atomic_json_dump,
    destroy_distributed,
    git_metadata,
    initialize_distributed,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--data-dir", default="data/tinystories_full")
    parser.add_argument("--output", default="results/c0_environment/preflight.json")
    parser.add_argument("--collective-mib", type=int, default=64)
    parser.add_argument("--collective-iters", type=int, default=10)
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero if a required check fails")
    return parser.parse_args()


def command_output(command: list[str], timeout: int = 60) -> dict:
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": process.returncode,
            "output": process.stdout.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "output": str(error)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_inventory() -> tuple[list[dict], dict, dict]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query["returncode"] == 0:
        for line in query["output"].splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "uuid": fields[1],
                        "name": fields[2],
                        "memory_total_mib": int(fields[3]),
                        "driver_version": fields[4],
                    }
                )
    topology = command_output(["nvidia-smi", "topo", "-m"])
    nvcc = command_output(["nvcc", "--version"])
    return gpus, topology, nvcc


def dataset_report(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    report = {
        "data_dir": str(data_dir),
        "manifest_exists": manifest_path.is_file(),
        "files": {},
    }
    if not manifest_path.is_file():
        return report
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["dataset_id"] = manifest.get("dataset_id")
    report["dataset_revision"] = manifest.get("dataset_revision")
    for split, filename in (("train", "train.bin"), ("validation", "val.bin")):
        path = data_dir / filename
        expected = manifest.get("splits", {}).get(split, {}).get("sha256")
        actual = sha256_file(path) if path.is_file() else None
        report["files"][split] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": expected is not None and expected == actual,
        }
    return report


def run_collective(env, size_mib: int, iterations: int) -> dict | None:
    if not env.distributed:
        return None
    elements = size_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    tensor = torch.empty(elements, dtype=torch.float32, device=env.device)
    expected = env.world_size * (env.world_size + 1) / 2
    for _ in range(3):
        tensor.fill_(env.rank + 1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    valid = True
    for _ in range(iterations):
        tensor.fill_(env.rank + 1)
        dist.all_reduce(tensor)
        valid = valid and bool(torch.all(tensor == expected))
    torch.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - started
    timing = torch.tensor(elapsed, device=env.device)
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    validity = torch.tensor(int(valid), device=env.device)
    dist.all_reduce(validity, op=dist.ReduceOp.MIN)
    if not env.master:
        return None
    worst_elapsed = float(timing)
    return {
        "backend": dist.get_backend(),
        "world_size": env.world_size,
        "payload_mib": size_mib,
        "iterations": iterations,
        "values_correct": bool(validity),
        "worst_rank_total_seconds": worst_elapsed,
        "worst_rank_mean_milliseconds": worst_elapsed * 1000 / iterations,
    }


def main() -> None:
    args = parse_args()
    launched = "RANK" in os.environ
    env = initialize_distributed(
        device="cuda" if launched else "auto",
        backend="nccl" if launched else "auto",
    )
    try:
        collective = run_collective(env, args.collective_mib, args.collective_iters)
        if env.distributed:
            dist.barrier()
        if not env.master:
            return

        gpus, topology, nvcc = gpu_inventory()
        data = dataset_report(Path(args.data_dir))
        disk = shutil.disk_usage(REPOSITORY_ROOT)
        memory_kib = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    memory_kib = int(line.split()[1])
                    break
        except OSError:
            pass

        fsdp2_available = False
        fsdp2_error = None
        try:
            from torch.distributed.fsdp import fully_shard  # noqa: F401
            from torch.distributed.checkpoint.state_dict import get_state_dict  # noqa: F401
            fsdp2_available = True
        except (ImportError, AttributeError) as error:
            fsdp2_error = str(error)

        names = {gpu["name"] for gpu in gpus}
        memories = {gpu["memory_total_mib"] for gpu in gpus}
        required_checks = {
            "expected_gpu_count": len(gpus) == args.expected_gpus,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_sees_expected_gpus": torch.cuda.device_count() == args.expected_gpus,
            "homogeneous_gpu_model": len(names) == 1,
            "homogeneous_gpu_memory": len(memories) == 1,
            "bf16_supported": (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "nccl_available": dist.is_nccl_available(),
            "train_data_hash_matches": data.get("files", {}).get("train", {}).get("sha256_matches", False),
            "validation_data_hash_matches": data.get("files", {}).get("validation", {}).get("sha256_matches", False),
            "deepspeed_installed": package_version("deepspeed") is not None,
            "fsdp2_available": fsdp2_available,
        }
        if launched:
            required_checks["collective_world_size"] = env.world_size == args.expected_gpus
            required_checks["nccl_all_reduce_correct"] = bool(
                collective and collective["values_correct"]
            )
        warnings = {
            "cuda_toolkit_nvcc_available": nvcc["returncode"] == 0,
            "at_least_50_gib_free_disk": disk.free >= 50 * 1024**3,
            "launched_distributed_collective": launched,
            "git_worktree_clean": not git_metadata()["dirty"],
        }
        failed = sorted(name for name, passed in required_checks.items() if not passed)
        warning_names = sorted(name for name, passed in warnings.items() if not passed)
        result = {
            "schema_version": 1,
            "stage": "C0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail" if failed else ("warn" if warning_names else "pass"),
            "expected_gpus": args.expected_gpus,
            "required_checks": required_checks,
            "warnings": warnings,
            "failed_required_checks": failed,
            "active_warnings": warning_names,
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total_bytes": memory_kib * 1024 if memory_kib else None,
                "disk_total_bytes": disk.total,
                "disk_free_bytes": disk.free,
            },
            "software": {
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
                "deepspeed": package_version("deepspeed"),
                "numpy": package_version("numpy"),
                "fsdp2_available": fsdp2_available,
                "fsdp2_error": fsdp2_error,
            },
            "gpus": gpus,
            "nvidia_smi_topology": topology,
            "nvcc": nvcc,
            "collective": collective,
            "dataset": data,
            "git": git_metadata(),
        }
        atomic_json_dump(result, args.output)
        print(json.dumps({
            "status": result["status"],
            "output": args.output,
            "failed_required_checks": failed,
            "active_warnings": warning_names,
        }, ensure_ascii=False, indent=2))
        if args.strict and failed:
            raise SystemExit(1)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
