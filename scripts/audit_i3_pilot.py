"""Independently recompute diagnostic capacity and the frozen arrival design."""
import argparse
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write,schedules
from serving.i3_audit import service_checks,source_checks
from serving.i2_workloads import digest


def audit(directory):
    meta=read(directory/'provenance.json');config=meta['config'];workload=read(directory/'workload.json')
    checks=dict(frozen_sources=source_checks(directory,meta),
        predetermined_SLO=config['latency_objectives_s']==dict(first_text=1.,e2e=8.,amortized_tpot=.1))
    capacities={}
    for variant in config['variants']:
        path=directory/f'pilot_{variant}'
        checks.update({variant+'_'+k:v for k,v in service_checks(path,config,workload,variant,'pilot').items()})
        requests=[r for r in read(path/'requests.json') if r['request_id'].startswith('load_')]
        send=[e['monotonic_ns'] for r in requests for e in r['events'] if e['event']=='http_send']
        finish=[e['monotonic_ns'] for r in requests for e in r['events'] if e['event']=='finished']
        seconds=(max(finish)-min(send))/1e9
        capacity=len(requests)/seconds;capacities[variant]=capacity
        stored=read(path/'capacity.json')
        checks[variant+'_capacity']=(len(requests)==48 and len(finish)==48 and stored['requests']==48
            and math.isclose(seconds,stored['seconds'],rel_tol=1e-12)
            and math.isclose(capacity,stored['requests_per_s'],rel_tol=1e-12))
    checks['frozen_schedule']=schedules(workload,capacities,config)==read(directory/'schedule.json')
    journal=read(directory/'processes.json')
    checks['two_clean_processes']=len(journal)==2 and [r['label'] for r in journal]==[f'pilot_{v}' for v in config['variants']] and all(
        r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    return dict(passed=all(checks.values()),checks=checks,capacities_rps=capacities,
        rates_rps=read(directory/'schedule.json')['rates_rps'],formal_performance=False,
        provenance_sha256=digest(directory/'provenance.json'),schedule_sha256=digest(directory/'schedule.json'),
        auditor_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    target=args.directory/'audit.json'
    if target.exists():raise ValueError('Preserve pilot audit')
    result=audit(args.directory);write(target,result)
    dest=args.directory/'audit_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),dest/Path(__file__).name)
    print('pilot audit passed:',result['passed'],'failed:',[k for k,v in result['checks'].items() if not v],flush=True)
    raise SystemExit(0 if result['passed'] else 1)
