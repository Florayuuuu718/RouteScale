"""Independent policy-presence and HTTP lifecycle audit for all seven engines."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest
from scripts.run_i2_service_gate import audit_service


def audit(directory):
    meta=json.loads((directory/'provenance.json').read_text());config=meta['config']
    checks=dict(frozen_sources=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items()))
    reports={}
    for name in config['variants']:
        path=directory/f'service_{name}';actual=audit_service(path,config)
        checks[name]=actual==json.loads((path/'audit.json').read_text()) and actual['passed']
        policy=[r for r in json.loads((path/'events.json').read_text()) if r['event']=='i2_shape_policy']
        checks[f'{name}_policy']=len(policy)==1 and policy[0]['name']==meta['policy'] and policy[0]['source_sha256']=={n:meta['source_sha256'][n] for n in ('serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py')}
        reports[name]=dict(passed=actual['passed'],checks=len(actual.get('checks',{})),audit_sha256=digest(path/'audit.json'))
    journal=json.loads((directory/'processes.json').read_text())
    checks['seven_clean_processes']=len(journal)==7 and all(r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    summary=json.loads((directory/'summary.json').read_text())
    checks['summary']=summary['HTTP_lifecycle_passed'] and summary['requests']==84 and summary['checks']==sum(r['checks'] for r in reports.values())
    return dict(passed=all(checks.values()),checks=checks,variants=reports,requests=84,scenario_checks=sum(r['checks'] for r in reports.values()),
        performance_allowed=False,source_sha256=digest(Path(__file__)),provenance_sha256=digest(directory/'provenance.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'independent_audit.json'
    if path.exists():raise ValueError('Preserve lifecycle audit')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('passed','requests','scenario_checks')}))
    raise SystemExit(0 if result['passed'] else 1)
