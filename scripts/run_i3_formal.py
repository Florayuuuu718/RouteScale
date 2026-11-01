"""Actual formal admission: both independent gates, immutable design, 18 services."""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import CONFIG,read,write
from serving.i2_workloads import digest
from scripts.run_i3_online import freeze,prerequisites


def gates(qualification,pilot,config):
    binding=prerequisites(qualification,pilot,config)
    audit=read(pilot/'audit.json')
    if not audit['passed'] or audit['auditor_sha256']!=digest(ROOT/'scripts/audit_i3_pilot.py'):
        raise ValueError('Current independent pilot gate required')
    if audit['provenance_sha256']!=digest(pilot/'provenance.json') or audit['schedule_sha256']!=digest(pilot/'schedule.json'):
        raise ValueError('Independent pilot binding changed')
    binding.update(pilot_audit_sha256=digest(pilot/'audit.json'),admitter_sha256=digest(Path(__file__)))
    return binding


def telemetry():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,temperature.gpu,clocks.sm,power.draw,pstate',
        '--format=csv,noheader'],text=True).strip()


def main():
    from scripts.run_i2_precision_followup import require_idle_gpu
    from scripts.run_i2_single_gpu import launch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    args=p.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Fresh simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require_idle_gpu();config=read(CONFIG);q=ROOT/args.qualification;pilot=ROOT/args.pilot
    gate=gates(q,pilot,config)
    directory=ROOT/config['results_dir']/args.attempt
    meta=freeze(directory,config,read(q/'workload.json'),'formal')
    for name in ['scripts/run_i3_formal.py','scripts/audit_i3_pilot.py','scripts/audit_i3_online.py','tests/test_i3_admission.py']:
        meta['source_sha256'][name]=digest(ROOT/name)
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    meta['artifact_bindings']=gate
    write(directory/'provenance.json',meta);write(directory/'gate.json',gate)
    shutil.copy2(pilot/'schedule.json',directory/'schedule.json')
    write(directory/'hardware.json',dict(start=telemetry(),boundary='before service startup / after process exit, not kernel-clock telemetry'))
    journal=[];states=[];rates=['low','near','overload'];variants=list(config['variants'])
    for run in range(1,4):
        order=rates if run==1 else list(reversed(rates)) if run==2 else rates[1:]+rates[:1]
        for index,rate in enumerate(order):
            policies=variants if run==1 or (run==3 and index%2==0) else list(reversed(variants))
            for variant in policies:
                if any(digest(ROOT/n)!=sha for n,sha in meta['source_sha256'].items()):
                    raise ValueError('Frozen formal sources changed')
                label=f'formal_{variant}_{rate}_run{run}'
                command=[sys.executable,'scripts/i3_service_worker.py','--directory',str(directory),'--label',label,
                    '--variant',variant,'--stage','formal','--rate',rate,'--run',str(run)]
                launch(command,label,config,args.attempt,directory,journal)
                if journal[-1]['gpu_after']!='0 MiB':raise ValueError('GPU not idle after process; preserve attempt')
                states.append(dict(label=label,telemetry=telemetry()))
                write(directory/'hardware_after_processes.json',states)


if __name__=='__main__':main()
