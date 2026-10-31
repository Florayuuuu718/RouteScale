"""Bounded, serial three-process diagnosis; original pressure results are frozen."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.backends.nanovllm import digest,write_result
from serving.h2_kv_diagnostic import DEFAULT_CONFIG
from serving.h2_runtime import now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    directory=ROOT/config["results_dir"]
    if any(directory.rglob("*.json")):
        raise SystemExit("Choose a fresh results_dir; never overwrite a diagnostic attempt")
    lock=(ROOT/"runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    python=str(ROOT/"runs/envs/nano-vllm/bin/python")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/"runs/cache/h_nanovllm/inductor"),
        TRITON_CACHE_DIR=str(ROOT/"runs/cache/h_nanovllm/triton"))
    logs=ROOT/"runs/logs/h_nanovllm"/directory.name
    logs.mkdir(parents=True,exist_ok=True)
    report=dict(schema_version=1,stage="H2",status="running",started_at_utc=now(),pid=os.getpid(),
        config_sha256=digest(args.config),completed_runs=0)
    try:
        for step in ("run1","run2","run3","audit"):
            report.update(current_step=step,updated_at_utc=now())
            write_result(report,directory/"run.json")
            command=[python,"-m","serving.h2_kv_diagnostic","--config",str(args.config)]
            if step!="audit":
                command += ["--run-id",step[-1],"--result",str(directory/"raw"/f"{step}.json")]
            print(f"KV negative-control diagnostic: {step}",flush=True)
            with (logs/f"{step}.log").open("x") as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            if step!="audit":
                report["completed_runs"]+=1
        report.update(status="pass",current_step="complete")
    except BaseException:
        report.update(status="fail",traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"]=now()
        write_result(report,directory/"run.json")


if __name__=="__main__":
    main()
