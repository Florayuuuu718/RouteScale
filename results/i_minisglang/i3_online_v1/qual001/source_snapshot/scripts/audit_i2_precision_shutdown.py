"""Verify the explicit teardown budget while retaining the external 10 s gate."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def audit(directory):
    read=lambda p:json.loads(p.read_text())
    meta=read(directory/'provenance.json');config=meta['config'];http=read(directory/'audit.json')
    checks=dict(HTTP_passed=http['HTTP_lifecycle_passed'],helper_source=digest(ROOT/'serving/i2_worker_shutdown.py')==meta['source_sha256']['serving/i2_worker_shutdown.py'])
    for mode in config['modes']:
        for variant in config['variants']:
            path=directory/mode/f'service_{variant}'
            events=read(path/'events.json');run=read(path/'run.json')
            policies=[r for r in events if r['event']=='i2_shutdown_policy']
            checks[f'{mode}_{variant}']=(len(policies)==1 and policies[0]['budgets']==[
                dict(signal='SIGINT',seconds=6.),dict(signal='SIGTERM',seconds=1.5),dict(signal='SIGKILL',seconds=.5)]
                and policies[0]['total_budget_s']==8 and policies[0]['external_shutdown_budget_s']==10
                and policies[0]['source_sha256']==meta['source_sha256']['serving/i2_worker_shutdown.py']
                and not run['forced_cleanup'] and not run['surviving_pids'] and run['shutdown_s']<=10)
    return dict(passed=all(checks.values()),checks=checks,external_shutdown_budget_s=10,phase_budget_total_s=8,
        HTTP_audit_sha256=digest(directory/'audit.json'),source_sha256=digest(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();output=args.directory/'shutdown_audit.json'
    if output.exists():raise ValueError('Preserve shutdown audit')
    result=audit(args.directory);output.write_text(json.dumps(result,indent=2)+'\n')
    snapshot=args.directory/'shutdown_audit_source_snapshot/scripts';snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
