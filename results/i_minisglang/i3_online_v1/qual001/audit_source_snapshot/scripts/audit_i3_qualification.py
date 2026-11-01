"""Reopen full-vocabulary tensors and actual HTTP lifecycle evidence for I3."""
import argparse
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest
from serving.i3_audit import service_checks,source_checks


def audit(directory):
    import torch
    torch.set_num_threads(4)
    from serving.i2_worker import error_metrics
    meta=read(directory/'provenance.json');config=meta['config'];workload=read(directory/'workload.json')
    checks=dict(frozen_sources=source_checks(directory,meta), unchanged_threshold=config['numerics']==dict(atol=.125,rtol=.02,max_relative_rmse=.005))
    expected={r['request_id']:r for r in workload['requests']+workload['warmup']}
    profiles={};vectors=0
    for variant in config['variants']:
        path=directory/f'qualification_{variant}'
        checks.update({f'{variant}_{k}':v for k,v in service_checks(path,config,workload,variant,'qualification').items()})
        report=read(directory/f'hf_{variant}.json');events=read(path/'events.json')
        assigned={e['external_id']:e['uid'] for e in events if e['event']=='frontend_new_user'}
        checks[variant+'_coverage']=set(report['requests'])==set(expected)
        errors=[];top1=0
        for rid,req in expected.items():
            r=report['requests'][rid]
            checks[variant+'_'+rid+'_hash']=(digest(ROOT/r['actual_path'])==r['actual_sha256'] and digest(ROOT/r['reference_path'])==r['reference_sha256'])
            actual=torch.load(ROOT/r['actual_path'],weights_only=True)
            reference=torch.load(ROOT/r['reference_path'],weights_only=True)
            tokens=[e['token_id'] for e in events if e['event']=='scheduler_token' and e['uid']==assigned[rid]]
            checks[variant+'_'+rid+'_identity']=(r['uid']==assigned[rid] and actual['input_ids']==req['input_ids'] and tokens==r['tokens']
                and actual['logits'].shape==reference.shape==(req['output_budget'],151936))
            recomputed=error_metrics(actual['logits'],reference,config['numerics'])
            checks[variant+'_'+rid+'_metrics']=(len(recomputed)==len(r['errors']) and all(
                all(a[k]==b[k] for k in ('step','finite','tolerance_violations','top1_equal','passed'))
                and all(math.isclose(a[k],b[k],rel_tol=1e-6,abs_tol=1e-9) for k in ('max_abs','relative_rmse'))
                for a,b in zip(recomputed,r['errors'],strict=True)))
            errors+=recomputed;top1+=sum(e['top1_equal'] for e in recomputed)
        profiles[variant]=dict(passed=all(e['passed'] for e in errors),vectors=len(errors),
            max_relative_rmse=max(e['relative_rmse'] for e in errors),max_abs=max(e['max_abs'] for e in errors),
            top1_equal_vectors=top1,failed_vectors=sum(not e['passed'] for e in errors))
        vectors+=len(errors)
        print('I3 independent qualification',variant,profiles[variant],flush=True)
    journal=read(directory/'processes.json')
    labels=[f'qualification_{v}' for v in config['variants']]+[f'hf_{v}' for v in config['variants']]
    checks['four_clean_processes']=(len(journal)==4 and [r['label'] for r in journal]==labels and all(
        r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal))
    passed=all(checks.values()) and all(p['passed'] for p in profiles.values())
    return dict(passed=passed,stage='I3_qualification',checks=checks,profiles=profiles,vectors=vectors,
        full_vocabulary=151936,performance_allowed=False,natural_generation_equivalence_claimed=False,
        provenance_sha256=digest(directory/'provenance.json'),auditor_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    target=args.directory/'audit.json'
    if target.exists():raise ValueError('Preserve qualification audit')
    result=audit(args.directory);write(target,result)
    dest=args.directory/'audit_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),dest/Path(__file__).name)
    print('qualification passed:',result['passed'],flush=True)
    raise SystemExit(0 if result['passed'] else 1)
