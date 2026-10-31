"""Resume H1 after verified artifact downloads; gate each GPU step on correctness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from serving.backends.nanovllm import DEFAULT_CONFIG, ROOT, digest, load_config, write_result


ARTIFACTS = [
    ("runs/downloads/torch-2.8.0+cu129-cp311-cp311-manylinux_2_28_x86_64.whl", 1240457889,
     "da19696fd75c4a2d5bc945242619143dfc4cbc3e3deead407f2946d1395c3608"),
    ("runs/downloads/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl", 706758467,
     "949452be657fa16687d0930933f032835951ef0892b37d2d53824d1a84dc97a8"),
    ("runs/downloads/nvidia_nccl_cu12-2.27.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", 322364134,
     "adf27ccf4238253e0b826bce3ff5fa532d65fc42322c8bfdfaf28024c0fbe039"),
    ("runs/downloads/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", 155531138,
     "7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467"),
    ("runs/models/Qwen3-0.6B/model.safetensors", 1503300328,
     "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-downloads", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / "runs/tmp/h1_runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An H1 runner is already active; inspect results/h_nanovllm/h1_run.json")
    config = load_config()
    result_path = ROOT / "results/h_nanovllm/h1_run.json"
    report = {"schema_version": 1, "stage": "H1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "functional_reproduction", "completed_steps": [],
        "entrypoint_sha256": digest(Path(__file__)),
        "config_sha256": digest(DEFAULT_CONFIG), "pid": os.getpid()}
    python = str(ROOT / config["environment"]["path"] / "bin/python")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT / "runs/cache/h_nanovllm/triton"))
    logs = ROOT / "runs/logs/h_nanovllm"
    logs.mkdir(parents=True, exist_ok=True)

    def run(name, command):
        report.update(status="running", current_step=name)
        write_result(report, result_path)
        print(f"H1: {name}", flush=True)
        with (logs / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True)
        report["completed_steps"].append(name)
        write_result(report, result_path)

    try:
        while True:
            pending = [name for name, size, _ in ARTIFACTS
                       if not (ROOT / name).exists() or (ROOT / name).stat().st_size != size]
            report.update(current_step="artifact_downloads", pending_artifacts=pending)
            write_result(report, result_path)
            if not pending:
                break
            if not args.wait_for_downloads:
                raise RuntimeError("Artifacts are incomplete; see pending_artifacts")
            time.sleep(10)
        for name, _, checksum in ARTIFACTS:
            if digest(ROOT / name) != checksum:
                raise RuntimeError(f"Artifact checksum mismatch: {name}")
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv must be available for the isolated environment")
        run("install_torch", [uv, "pip", "install", "--python", python,
            *[str(ROOT / name) for name, _, _ in ARTIFACTS if name.endswith(".whl")]])
        run("environment_check", [uv, "pip", "check", "--python", python])
        # HF recognizes the already verified local LFS file and writes revision metadata.
        run("model_metadata", [python, "-c",
            "from huggingface_hub import snapshot_download; snapshot_download("
            f"{config['model']['repo_id']!r}, revision={config['model']['revision']!r}, "
            f"local_dir={str(ROOT / config['model']['path'])!r})"])
        for step, module in [("state", "verify_nanovllm_state"),
                             ("attention", "verify_nanovllm_attention"),
                             ("generation", "reproduce_nanovllm")]:
            run(step, [python, "-m", "serving." + module])
        report.update(status="pass", current_step="complete")
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(report, result_path)
    print(f"H1 complete: {result_path}", flush=True)


if __name__ == "__main__":
    main()
