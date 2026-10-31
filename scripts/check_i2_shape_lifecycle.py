"""Recheck seven HTTP profiles under the numerically qualified common policy."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification',type=Path);parser.add_argument('--attempt')
    parser.add_argument('--worker',type=Path);parser.add_argument('--variant')
    args=parser.parse_args()
    if args.worker:
        from scripts.run_i2_service_gate import worker
        meta=json.loads((args.worker/'provenance.json').read_text())
        for n,sha in meta['source_sha256'].items():
            if digest(ROOT/n)!=sha:raise ValueError('Frozen service sources changed')
        return worker(meta['config'],args.worker.name,args.variant)
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qualification=json.loads((args.qualification/'audit.json').read_text())
    if not qualification['qualification_passed']:raise ValueError('All held-out numerical profiles must qualify first')
    base=json.loads((args.qualification/'provenance.json').read_text())
    config=dict(base['config'],service_launcher='scripts/launch_i2_shape_service.py',
        results_dir='results/i_minisglang/i2_shape_qualification_v1',runtime_dir='runs/logs/i_minisglang/i2_shape_qualification_v1')
    directory=ROOT/config['results_dir']/args.attempt;directory.mkdir(parents=True,exist_ok=False)
    names=['scripts/check_i2_shape_lifecycle.py','scripts/launch_i2_shape_service.py','scripts/launch_i2_service.py',
        'scripts/run_i2_service_gate.py','scripts/run_i2_single_gpu.py','serving/i2_numerical_policy.py',
        'serving/i2_linear_tile_policy.py','serving/minisgl_observer.py','serving/online_client.py',
        'scripts/run_i1_lifecycle.py','scripts/run_i1_repaired.py']
    meta=dict(config=config,source_sha256={n:digest(ROOT/n) for n in names},
        qualification_sha256=digest(args.qualification/'audit.json'),policy='fi_prefill_no_split_linear512',performance_allowed=False)
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    from scripts.run_i2_service_gate import audit_service
    from scripts.run_i1_lifecycle import events
    journal=[];results={}
    for variant in config['variants']:
        launch([sys.executable,__file__,'--worker',str(directory),'--variant',variant],
            f'service_{variant}',config,args.attempt,directory,journal)
        output=directory/f'service_{variant}';actual=audit_service(output,config)
        run=json.loads((output/'run.json').read_text())
        policy=[r for r in events(ROOT/run['runtime_dir']) if r['event']=='i2_shape_policy']
        valid=len(policy)==1 and policy[0]['name']==meta['policy'] and policy[0]['source_sha256']=={
            n:meta['source_sha256'][n] for n in ('serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py')}
        if not actual['passed'] or not valid:raise ValueError('Policy lifecycle validation failed')
        results[variant]=dict(passed=True,checks=len(actual['checks']),policy_observed=True,audit_sha256=digest(output/'audit.json'))
    (directory/'summary.json').write_text(json.dumps(dict(HTTP_lifecycle_passed=True,variants=results,
        checks=sum(r['checks'] for r in results.values()),requests=84,performance_allowed=False),indent=2)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
