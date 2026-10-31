"""Freeze and run two serial GPU traces on the unchanged I2 native tree."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import CONFIG,digest
from scripts.prepare_i2_minisglang import prepare_i2
from scripts.run_i2_single_gpu import launch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads(CONFIG.read_text())
    directory=ROOT/'results/i_minisglang/i2_numerical_trace_v1'/args.attempt
    validation=ROOT/'runs/validation/i2_numerical_trace_v1'/args.attempt
    directory.mkdir(parents=True,exist_ok=False);validation.mkdir(parents=True,exist_ok=False)
    names=['serving/i2_layer_trace.py','serving/i2_trace_worker.py','scripts/run_i2_layer_trace.py',
        'serving/i2_runtime.py','serving/i2_workloads.py','serving/i2_worker.py','scripts/run_i2_single_gpu.py',
        'scripts/prepare_i2_minisglang.py','scripts/prepare_i1_lifecycle.py','scripts/run_i1_repaired.py',
        'scripts/run_i1_lifecycle.py','docs/i2_layer_trace_protocol.md',str(CONFIG.relative_to(ROOT))]
    series=prepare_i2()
    meta=dict(config=config,source_sha256={n:digest(ROOT/n) for n in names},series=series,
        validation_dir=str(validation.relative_to(ROOT)),workload_sha256=digest(ROOT/config['workload_path']),
        model_sha256={p.name:digest(p) for p in (ROOT/config['model_path']).iterdir() if p.is_file()},
        performance_allowed=False,paid_resources_used=False)
    for name in names:
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(ROOT/config['workload_path'],directory/'workload.json')
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    runtime=dict(config,runtime_dir='runs/logs/i_minisglang/i2_numerical_trace_v1',subprocess_timeout_s=300)
    journal=[]
    for variant in ('naive_full','radix_full'):
        launch([sys.executable,'-m','serving.i2_trace_worker','--directory',str(directory),'--variant',variant],
            variant,runtime,args.attempt,directory,journal)


if __name__=='__main__':main()
