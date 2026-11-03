"""Serial I3 orchestration: prepare, diagnostic pilot, qualification, 18 HTTP runs."""
import argparse
import fcntl
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import CONFIG,read,write,make_workload,schedules
from serving.i2_workloads import digest,token_hash


def prerequisites(qualification,pilot,config):
    from serving.i3_audit import source_checks
    q=read(qualification/'audit.json');qm=read(qualification/'provenance.json');pm=read(pilot/'provenance.json')
    if not q['passed'] or q['auditor_sha256']!=digest(ROOT/'scripts/audit_i3_qualification.py'):
        raise ValueError('Current independent I3 qualification required')
    if q['provenance_sha256']!=digest(qualification/'provenance.json') or qm['config']!=config or pm['config']!=config:
        raise ValueError('Gate config/provenance changed')
    if not source_checks(qualification,qm) or not source_checks(pilot,pm):
        raise ValueError('Qualified/pilot sources changed')
    if qm['workload_sha256']!=pm['workload_sha256']:
        raise ValueError('Qualified workload differs from pilot')
    capacities={v:read(pilot/f'pilot_{v}/capacity.json')['requests_per_s'] for v in config['variants']}
    if schedules(read(pilot/'workload.json'),capacities,config)!=read(pilot/'schedule.json'):
        raise ValueError('Frozen pilot schedule changed')
    return dict(qualification_directory=str(qualification.relative_to(ROOT)),pilot_directory=str(pilot.relative_to(ROOT)),
        qualification_audit_sha256=digest(qualification/'audit.json'),pilot_provenance_sha256=digest(pilot/'provenance.json'),
        schedule_sha256=digest(pilot/'schedule.json'),performance_allowed=True)


def freeze(directory,config,workload,stage):
    from scripts.prepare_i2_minisglang import validate
    acceptance=ROOT/config['I2_acceptance'];prior=read(acceptance)
    if not prior['local_single_gpu_complete']:
        raise ValueError('I2 local acceptance required')
    names=set()
    for scope in ('qual001','http004','perf001'):
        previous=read(acceptance.parent/scope/'provenance.json')
        names.update(previous['source_sha256'])
        if any(digest(ROOT/n)!=s for n,s in previous['source_sha256'].items()):
            raise ValueError('Accepted I2 sources changed')
    names.update(['config/serving/i3_online.json','docs/i3_online_protocol.md',
        'scripts/run_i3_online.py','scripts/i3_service_worker.py','scripts/i3_reference_worker.py',
        'scripts/launch_i3_service.py','scripts/audit_i3_qualification.py',
        'serving/i3_workloads.py','serving/i3_metrics.py','serving/i3_observer.py','serving/i3_audit.py',
        'tests/test_i3_online.py'])
    names.update(['docs/i3_online_protocol_v2.md','serving/i3_precision_v2.py','tests/test_i3_precision.py'])
    directory.mkdir(parents=True,exist_ok=False)
    write(directory/'workload.json',workload)
    meta=dict(stage=stage,config=config,config_sha256=token_hash(config),workload_sha256=digest(directory/'workload.json'),
        source_sha256={n:digest(ROOT/n) for n in sorted(names)},series=validate(ROOT/config['checkout']),
        model_sha256={p.name:digest(p) for p in (ROOT/config['model_path']).iterdir() if p.is_file()},
        I2_acceptance_sha256=digest(acceptance),paid_resources_used=False,performance_allowed=stage=='formal')
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    write(directory/'provenance.json',meta)
    return meta


def main():
    from scripts.run_i2_precision_followup import require_idle_gpu
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','pilot','qualification','formal'],required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--source',type=Path)
    p.add_argument('--qualification',type=Path);p.add_argument('--pilot',type=Path)
    args=p.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Fresh simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_idle_gpu();config=read(CONFIG)
    if args.stage=='prepare':
        workload=make_workload(config)
    else:
        if args.source is None:raise ValueError('Frozen workload source required')
        workload=read(ROOT/args.source/'workload.json')
    gate=None
    if args.stage=='formal':
        if args.qualification is None or args.pilot is None:raise ValueError('Explicit gate bindings required')
        gate=prerequisites(ROOT/args.qualification,ROOT/args.pilot,config)
    directory=ROOT/config['results_dir']/args.attempt
    meta=freeze(directory,config,workload,args.stage)
    if gate:
        if digest(directory/'workload.json')!=read(ROOT/args.qualification/'provenance.json')['workload_sha256']:
            raise ValueError('Formal workload not qualified')
        shutil.copy2(ROOT/args.pilot/'schedule.json',directory/'schedule.json')
        write(directory/'gate.json',gate)
    if args.stage=='prepare':return
    from scripts.run_i2_single_gpu import launch
    journal=[]
    def call(label,command):
        if any(digest(ROOT/n)!=sha for n,sha in meta['source_sha256'].items()):
            raise ValueError('Frozen source changed during run')
        launch([sys.executable,*command],label,config,args.attempt,directory,journal)
        if journal[-1]['gpu_after']!='0 MiB':raise ValueError('GPU failed to return to idle; preserve attempt')
    def service(variant,rate='',run=0):
        label=f'{args.stage}_{variant}'+(f'_{rate}_run{run}' if rate else '')
        call(label,['scripts/i3_service_worker.py','--directory',str(directory),'--label',label,
            '--variant',variant,'--stage',args.stage,'--rate',rate,'--run',str(run)])
    if args.stage in ('pilot','qualification'):
        for variant in config['variants']:service(variant)
        if args.stage=='qualification':
            for variant in config['variants']:
                call(f'hf_{variant}',['scripts/i3_reference_worker.py','--directory',str(directory),'--variant',variant])
        else:
            capacities={v:read(directory/f'pilot_{v}/capacity.json')['requests_per_s'] for v in config['variants']}
            write(directory/'schedule.json',schedules(workload,capacities,config))
            print('Frozen pilot design',read(directory/'schedule.json')['rates_rps'],flush=True)
    else:
        rates=['low','near','overload'];variants=list(config['variants'])
        for run in range(1,4):
            order=rates if run==1 else list(reversed(rates)) if run==2 else rates[1:]+rates[:1]
            for index,rate in enumerate(order):
                policies=variants if run==1 or (run==3 and index%2==0) else list(reversed(variants))
                for variant in policies:service(variant,rate,run)


if __name__=='__main__':main()
