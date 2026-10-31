"""Isolate the second observed divergence using the exact captured BF16 input."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def worker(directory):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from serving.i2_attention_reference import tensor_metrics
    from serving.i2_linear_tile_policy import tiled_linear
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    meta=json.loads((directory/'provenance.json').read_text())
    if not all(digest(ROOT/n)==sha for n,sha in meta['source_sha256'].items()):raise ValueError('Frozen probe sources changed')
    if not all(digest(ROOT/r['path'])==r['sha256'] for r in meta['inputs'].values()):raise ValueError('Captured inputs changed')
    if digest(ROOT/meta['checkpoint'])!=meta['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    traces={v:torch.load(ROOT/r['path'],mmap=True,weights_only=True) for v,r in meta['inputs'].items()}
    a,b=traces['naive_full'],traces['radix_full']
    assert torch.equal(a['ops']['0.silu_mul'],b['ops']['0.silu_mul'])
    with safe_open(ROOT/meta['checkpoint'],framework='pt',device='cpu') as handle:
        weight=handle.get_tensor('model.layers.0.mlp.down_proj.weight').cuda()
    x=a['ops']['0.silu_mul'].cuda()
    with torch.inference_mode():
        padded=F.pad(x,(0,0,384,0))
        full=F.linear(padded,weight)[-128:]
        short=F.linear(x,weight)
        controlled=tiled_linear(x,lambda value:F.linear(value,weight))
        math32=F.linear(x.float(),weight.float())
        output={n:t.cpu() for n,t in dict(full512=full,short128=short,controlled512=controlled,math32=math32).items()}
        path=ROOT/meta['validation_dir']/'outputs.pt';torch.save(output,path)
        record=dict(evidence_class='same_input_BF16_down_projection_geometry',performance_allowed=False,
            full_reproduces_trace=tensor_metrics(full.cpu(),a['ops']['0.down_proj']),
            short_reproduces_trace=tensor_metrics(short.cpu(),b['ops']['0.down_proj']),
            full_vs_short=tensor_metrics(short,full),full_vs_controlled=tensor_metrics(controlled,full),
            full_vs_math32=tensor_metrics(full,math32),short_vs_math32=tensor_metrics(short,math32),
            input_shape=list(x.shape),weight_shape=list(weight.shape),output=dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
        (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path);parser.add_argument('--name',default='downproj001');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    trace_meta=json.loads((args.trace/'provenance.json').read_text());config=trace_meta['config']
    if not json.loads((args.trace/'audit.json').read_text())['localization_valid']:raise ValueError('Valid trace first')
    directory=args.trace/args.name;directory.mkdir(parents=True,exist_ok=False)
    validation=ROOT/'runs/validation/i2_numerical_trace_v1'/args.trace.name/args.name;validation.mkdir(parents=True,exist_ok=False)
    names=['scripts/probe_i2_down_projection.py','serving/i2_attention_reference.py','serving/i2_linear_tile_policy.py','docs/i2_shape_policy_protocol.md']
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    checkpoint=ROOT/config['model_path']/'model.safetensors'
    meta=dict(source_sha256={n:digest(ROOT/n) for n in names},validation_dir=str(validation.relative_to(ROOT)),
        inputs={v:json.loads((args.trace/f'{v}.json').read_text())['tensors']['trace'] for v in ('naive_full','radix_full')},
        trace_audit_sha256=digest(args.trace/'audit.json'),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),performance_allowed=False)
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    runtime=dict(config,runtime_dir='runs/logs/i_minisglang/i2_numerical_trace_v1',subprocess_timeout_s=180)
    launch([sys.executable,__file__,'--worker',str(directory)],args.name,runtime,args.trace.name,directory,[])


if __name__=='__main__':main()
