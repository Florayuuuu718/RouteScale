"""Audit I2 HTTP lifecycle evidence without granting a performance gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest
from scripts.run_i2_service_gate import audit_service


def audit(directory):
    meta=json.loads((directory/'provenance.json').read_text())
    config=meta['config']
    rows=json.loads((directory/'processes.json').read_text())
    stored=json.loads((directory/'lifecycle_diagnostic.json').read_text())
    checks={
        'frozen_sources':all(digest(directory/'source_snapshot'/n)==value for n,value in meta['source_sha256'].items()),
        'model_hashes':all(digest(ROOT/config['model_path']/n)==value for n,value in meta['model_sha256'].items()),
        'workload_hash':digest(directory/'workload.json')==meta['workload_sha256'],
        'complete_matrix':len(rows)==len(config['variants']) and {r['label'] for r in rows}=={f'service_{v}' for v in config['variants']},
        'processes_clean':all(r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in rows),
        'no_timing_gate':stored['performance_allowed'] is False and not (directory/'service_gate.json').exists(),
        'no_benchmark_processes':not any(r['label'].startswith('benchmark') for r in rows),
    }
    results={}
    for variant in config['variants']:
        path=directory/f'service_{variant}'
        actual=audit_service(path,config)
        checks[variant]=actual['passed'] and actual==json.loads((path/'audit.json').read_text())
        results[variant]=dict(passed=actual['passed'],checks=len(actual.get('checks',{})),audit_sha256=digest(path/'audit.json'))
    checks['controller_summary']=stored['passed']==all(r['passed'] for r in results.values()) and stored['variants']==results
    return dict(passed=all(checks.values()),checks=checks,variants=results,
        evidence_class='independent_HTTP_lifecycle_diagnostic_audit',
        numerical_acceptance=False,performance_allowed=False,local_single_gpu_complete=False,
        scenario_checks=sum(r['checks'] for r in results.values()),requests=12*len(results),
        source_sha256=digest(Path(__file__)),provenance_sha256=digest(directory/'provenance.json'),
        summary_sha256=digest(directory/'lifecycle_diagnostic.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    result=audit(args.directory)
    path=args.directory/'independent_audit.json'
    if args.check_only:
        if result!=json.loads(path.read_text()):raise ValueError('Independent audit changed')
    elif path.exists():raise ValueError('Preserve audit')
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','scenario_checks','requests','performance_allowed')}))
    raise SystemExit(0 if result['passed'] else 1)
