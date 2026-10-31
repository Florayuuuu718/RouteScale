"""Retryable HTTP/timing orchestration with explicit artifact bindings.

The accepted numerical controller and compute config remain byte-for-byte
unchanged. Artifact locations are selected separately so a failed HTTP attempt
is preserved without repeating unchanged numerical experiments.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest
from scripts.run_i2_precision import read,write,load_config,freeze,check_sources


def prerequisites(config,bindings):
    q=ROOT/bindings['qualification_directory'];l=ROOT/bindings['lifecycle_directory']
    qa,qm=read(q/'audit.json'),read(q/'provenance.json')
    la,lm=read(l/'audit.json'),read(l/'provenance.json')
    sa=read(l/'shutdown_audit.json')
    if not qa['qualification_passed'] or not la['HTTP_lifecycle_passed'] or not sa['passed']:
        raise ValueError('Numerical, HTTP and shutdown gates required')
    if lm['qualification_audit_sha256']!=digest(q/'audit.json') or sa['HTTP_audit_sha256']!=digest(l/'audit.json'):
        raise ValueError('Gate binding changed')
    for path,meta,audit in ((q,qm,qa),(l,lm,la)):
        if meta['config']!=config or audit['provenance_sha256']!=digest(path/'provenance.json'):
            raise ValueError('Qualified config/provenance changed')
        check_sources(meta)
        if digest(path/'workload.json')!=meta['workload_sha256']:
            raise ValueError('Qualified input changed')
        if audit['auditor_sha256']!=digest(ROOT/'scripts/audit_i2_precision.py'):
            raise ValueError('Gate auditor changed')
    if sa['source_sha256']!=digest(ROOT/'scripts/audit_i2_precision_shutdown.py'):
        raise ValueError('Shutdown auditor changed')
    for name,sha in qm['model_sha256'].items():
        if digest(ROOT/config['model_path']/name)!=sha:
            raise ValueError('Qualified checkpoint changed')
    return dict(performance_allowed=True,qualification_directory=str(q.relative_to(ROOT)),
        lifecycle_directory=str(l.relative_to(ROOT)),qualification_audit_sha256=digest(q/'audit.json'),
        lifecycle_audit_sha256=digest(l/'audit.json'),shutdown_audit_sha256=digest(l/'shutdown_audit.json'),modes=config['modes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['http','performance'],required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--lifecycle-directory',type=Path)
    args=parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Fresh simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=load_config();q=ROOT/config['qualification_directory']
    directory=ROOT/config['results_dir']/args.attempt
    lifecycle=(ROOT/args.lifecycle_directory).resolve() if args.lifecycle_directory else directory
    bindings=dict(qualification_directory=str(q.relative_to(ROOT)),lifecycle_directory=str(lifecycle.relative_to(ROOT)))
    qa=read(q/'audit.json')
    if not qa['qualification_passed']:raise ValueError('Numerical qualification first')
    gate=prerequisites(config,bindings) if args.stage=='performance' else None
    suites=read(q/'workload.json')
    workload=suites if args.stage=='http' else suites['suites'][0]['workload']
    meta=freeze(directory,config,args.stage,workload)
    meta.update(artifact_bindings=bindings,qualification_audit_sha256=digest(q/'audit.json'))
    extra=['scripts/run_i2_precision_followup.py','serving/i2_worker_shutdown.py','tests/test_i2_worker_shutdown.py']
    if args.stage=='performance':extra+=['scripts/audit_i2_precision_final.py','scripts/audit_i2_precision_shutdown.py']
    for name in extra:
        meta['source_sha256'][name]=digest(ROOT/name)
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(directory/'provenance.json',meta)
    if gate:write(directory/'gate.json',gate)
    from scripts.run_i2_single_gpu import launch
    journal=[]
    def call(kind,mode,variant,run=0):
        label=f'{kind}_{mode}_{variant}'+(f'_run{run}' if run else '')
        launch([sys.executable,'scripts/run_i2_precision.py','--worker',str(directory),'--kind',kind,
            '--mode',mode,'--variant',variant,'--run',str(run)],label,config,args.attempt,directory,journal)
    if args.stage=='http':
        for mode in config['modes']:
            for variant in config['variants']:call('service',mode,variant)
    else:
        names=list(config['variants'])
        for run in range(1,4):
            order=names if run==1 else list(reversed(names)) if run==2 else names[3:]+names[:3]
            for index,variant in enumerate(order):
                modes=config['modes'] if run==1 or (run==3 and index%2==0) else list(reversed(config['modes']))
                for mode in modes:call('benchmark',mode,variant,run)


if __name__=='__main__':main()
