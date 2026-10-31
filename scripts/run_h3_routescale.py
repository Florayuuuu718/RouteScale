"""Isolated offline preparation and three serial H3 functional GPU processes."""

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
from serving.h3_prepare import DEFAULT_CONFIG
from serving.h2_runtime import now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    directory=ROOT/config["results_dir"]
    if any(directory.rglob("*.json")):
        raise SystemExit("Select a new results_dir; never overwrite an H3 attempt")
    lock=(ROOT/"runs/tmp/h2_runner.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",PYTHONUNBUFFERED="1",
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/"runs/cache/h3_routescale/inductor"),
        TRITON_CACHE_DIR=str(ROOT/"runs/cache/h3_routescale/triton"))
    python=str(ROOT/"runs/envs/nano-vllm/bin/python")
    steps=[("prepare",[str(ROOT/".venv/bin/python"),"-m","serving.h3_prepare","--config",str(args.config)])]
    steps += [(v,[python,"-m","serving.verify_h3_routescale","--config",str(args.config),"--variant",v]) for v in config["variants"]]
    steps.append(("compare",[python,"-m","serving.verify_h3_routescale","--config",str(args.config)]))
    logs=ROOT/"runs/logs/h_nanovllm"/directory.name
    logs.mkdir(parents=True,exist_ok=True)
    report=dict(schema_version=1,stage="H3",status="running",pid=os.getpid(),started_at_utc=now(),
        config_sha256=digest(args.config),completed_steps=[])
    try:
        for name,command in steps:
            report.update(current_step=name,updated_at_utc=now())
            write_result(report,directory/"run.json")
            print(f"H3 RouteScale: {name}",flush=True)
            with (logs/f"{name}.log").open("x") as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            report["completed_steps"].append(name)
        report.update(status="pass",current_step="complete")
    except BaseException:
        report.update(status="fail",traceback=traceback.format_exc())
        raise
    finally:
        report["updated_at_utc"]=now()
        write_result(report,directory/"run.json")


if __name__=="__main__":
    main()
