"""Plan future TP runs, execute bounded local probes, or run on an existing host.

This tool never rents hosts. TP2/4 are not accepted by local TP1 success.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import read,sha,write
from serving.tp_preparation import CONFIG,inventory,jobs,launch,load_plan,numerical_gate,require_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['plan','collective','local-smoke','qualify','benchmark','trace'],required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--world',type=int,choices=[1,2,4],default=1)
    args=p.parse_args();c=load_plan();directory=args.directory.resolve()
    if args.stage=='plan':
        if directory.exists():raise ValueError('Fresh plan output required')
        write(directory/'plan.json',dict(config=c,qualification=jobs(c,'qualify'),benchmark=jobs(c,'benchmark'),
            config_sha256=sha(CONFIG),paid_resources_authorized=False,multigpu_accepted=False));return
    lock_path=ROOT/'runs/tmp/h2_runner.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ('benchmark','trace'):numerical_gate(directory,c)
        if args.stage in ('local-smoke','collective') and directory.exists():raise ValueError('Preserve probe directory')
        require_gpus(args.world if args.stage=='collective' else 1 if args.stage=='local-smoke' else 4)
        from serving.delivery import doctor,load
        doctor(load())
        nano=ROOT/c['frameworks']['nano']['checkout']
        import subprocess
        if (subprocess.check_output(['git','-C',str(nano),'rev-parse','HEAD'],text=True).strip()!=c['frameworks']['nano']['commit']
            or subprocess.check_output(['git','-C',str(nano),'status','--porcelain'],text=True).strip()):
            raise ValueError('Pinned clean nano-vLLM checkout required')
        directory.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            MINISGL_DISABLE_OVERLAP_SCHEDULING='1',CUDA_HOME=str(ROOT/'runs/toolchains/cuda-13.0-i1'),
            PATH=str(ROOT/'runs/envs/mini-sglang/bin')+':'+str(ROOT/'runs/toolchains/cuda-13.0-i1/bin')+':'+os.environ['PATH'],
            TORCHINDUCTOR_CACHE_DIR=str(ROOT/'runs/cache/tp_prepare/inductor'),TRITON_CACHE_DIR=str(ROOT/'runs/cache/tp_prepare/triton'))
        journal=read(directory/'processes.json') if (directory/'processes.json').exists() else []
        def call(label,command,world=1):
            require_gpus(world);child_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,range(world))))
            result=launch(command,directory/'logs'/f'{label}.log',c['subprocess_timeout_s'],child_env)
            result.update(label=label,gpu_after=require_gpus(world));journal.append(result);write(directory/'processes.json',journal)
            print('TP completed:',label,flush=True)
        mini=str(ROOT/c['frameworks']['mini']['environment']/'bin/python')
        if args.stage=='collective':
            inventory(directory/'host')
            call(f'collective_tp{args.world}',[mini,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={args.world}',
                'scripts/tp_worker.py','--mode','collective','--directory',str(directory)],args.world);return
        if args.stage in ('local-smoke','qualify'):
            if not (directory/'workload.json').exists():
                inventory(directory/'host')
                call('fixture',[mini,'scripts/tp_worker.py','--mode','fixture','--directory',str(directory)])
                call('HF_FP32',[mini,'scripts/tp_worker.py','--mode','reference','--directory',str(directory)])
                names=['config/delivery/tp_matrix.json','scripts/prepare_serving_tp.py','scripts/tp_worker.py','scripts/audit_serving_tp.py',
                    'serving/tp_preparation.py','serving/tp_precision.py','serving/tp_mini_runtime.py','serving/tp_nano_runtime.py']
                names += list(read(ROOT/'config/delivery/qualified_sources.json'))
                names += ['serving/h2_numerical_policy.py','serving/h2_prefix_probe.py','config/delivery/model.json',
                    'config/delivery/qualified_sources.json','config/serving/h1_nanovllm.json','docs/delivery_protocol.md',
                    'tests/test_tp_head_gather.py','tests/delivery/test_tp_preparation.py']
                names += [str(p.relative_to(ROOT)) for p in nano.rglob('*.py')]
                names += [str(p.relative_to(ROOT)) for p in (ROOT/c['frameworks']['mini']['checkout']/'python').rglob('*.py')]
                write(directory/'provenance.json',dict(config_sha256=sha(CONFIG),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},
                    model_manifest_sha256=sha(ROOT/c['model_manifest']),workload_sha256=sha(directory/'workload.json')))
                import shutil
                for name in sorted(set(names)):
                    target=directory/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        selected=jobs(c,'qualify' if args.stage=='local-smoke' else args.stage)
        if args.stage=='local-smoke':selected=[j for j in selected if j['world']==1]
        for j in selected:
            python=str(ROOT/c['frameworks'][j['framework']]['environment']/'bin/python')
            command=[python,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={j["world"]}',
                'scripts/tp_worker.py','--directory',str(directory),'--mode',j['stage'],'--framework',j['framework'],
                '--variant',j['variant'],'--label',j['label']]
            call(j['label'],command,j['world'])


if __name__=='__main__':main()
