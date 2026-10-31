"""Independent full-vocabulary audit of held-out I2 policy qualification."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest,validate_workload,token_hash


def audit(directory):
    import torch
    torch.set_num_threads(4)
    meta=json.loads((directory/'provenance.json').read_text());config=meta['config']
    workload=json.loads((directory/'workload.json').read_text())
    checks=dict(frozen_sources=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items()),
        workload_hash=digest(directory/'workload.json')==meta['workload_sha256'],
        model_hashes=all(digest(ROOT/config['model_path']/n)==sha for n,sha in meta['model_sha256'].items()),
        exact_scope=meta['reference_scope']=='within_native_BF16_strategy_invariance' and meta['performance_allowed'] is False)
    expected_ids=set();cases={}
    for suite in workload['suites']:
        validate_workload(suite['workload'],suite['config'])
        for case in suite['workload']['cases']:
            cases[(suite['name'],case['case_id'])]=(suite,case)
            for phase in ('cold','warm'):
                for r in case['waves'][phase]:
                    expected_ids.add(r['request_id'])
                    if r['continuation_ids']!=list(reversed(r['input_ids'][-64:])):raise ValueError('Continuation mismatch')
    checks['heldout_spec']=len(workload['suites'])==2 and len(expected_ids)==28 and workload['suites'][1]['config']['seed']==20261101 and workload['suites'][1]['config']['prompt_tokens']==768 and workload['suites'][1]['config']['batch_sizes']==[1,2]
    ref_record=json.loads((directory/'reference.json').read_text())
    checks['reference_hash']=digest(ROOT/ref_record['tensor']['path'])==ref_record['tensor']['sha256'] and ref_record['reference'] and ref_record['variant']=='naive_full'
    reference=torch.load(ROOT/ref_record['tensor']['path'],mmap=True,weights_only=True)
    checks['reference_coverage']=set(reference)==expected_ids and all(t.shape==(64,151936) and bool(torch.isfinite(t).all()) for t in reference.values())
    reports={};missing=[];vectors=0
    policy=config['numerics']
    for variant,settings in config['variants'].items():
        path=directory/f'{variant}.json'
        if not path.exists():
            missing.append(variant);continue
        record=json.loads(path.read_text());tensor=ROOT/record['tensor']['path']
        checks[f'{variant}_hash']=digest(tensor)==record['tensor']['sha256']
        target=torch.load(tensor,mmap=True,weights_only=True)
        checks[f'{variant}_coverage']=set(target)==expected_ids and set(record['errors'])==expected_ids and all(t.shape==(64,151936) for t in target.values())
        checks[f'{variant}_capture']=record['captured_buckets']==settings['graph_bs'] and record['performance_allowed'] is False
        failures=[];exact=0;maximum=0.;metrics_consistent=True
        for rid in sorted(expected_ids):
            a,b=target[rid],reference[rid]
            for step in range(64):
                x,y=a[step].float(),b[step].float()
                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                relative=float((x-y).norm()/y.norm().clamp_min(1e-12)) if finite else None
                violations=int((~torch.isclose(x,y,atol=policy['atol'],rtol=policy['rtol'])).sum())
                passed=finite and violations==0 and relative<=policy['max_relative_rmse']
                old=record['errors'][rid][step]
                metrics_consistent &= old['passed']==passed and old['finite']==finite and (math.isclose(relative,old['relative_rmse'],abs_tol=1e-8,rel_tol=1e-5) if finite else old['relative_rmse'] is None)
                exact+=torch.equal(x,y);vectors+=1
                if finite:maximum=max(maximum,relative)
                if not passed:failures.append(dict(request_id=rid,step=step,finite=finite,relative_rmse=relative,violations=violations))
        checks[f'{variant}_metrics_consistent']=bool(metrics_consistent) and record['within_native_passed']==(not failures)
        seen=set();work_ok=True
        for row in record['rows']:
            key=(row['suite'],row['case_id'],row['phase']);seen.add(key)
            suite,case=cases[key[:2]]
            expected_hit=suite['config']['shared_prefix_tokens'] if row['phase']=='warm' and case['prefix_mode']=='shared' and settings['cache']=='radix' else 0
            work_ok &= len(row['requests'])==case['batch_size'] and all(r['output_tokens']==64 and r['free_count']==1 and r['reason']=='length' for r in row['requests'])
            work_ok &= row['computed_prefill_tokens']==sum(r['extend'] for b in row['batches'] if b['phase']=='prefill' for r in b['requests'])
            work_ok &= row['computed_prefill_tokens']==case['batch_size']*suite['config']['prompt_tokens']-sum(r['cached_tokens'] for r in row['requests'])
            if row['phase']=='warm':work_ok &= all(r['cached_tokens']==expected_hit for r in row['requests'])
            released=row['released']
            work_ok &= not released['pending'] and not released['running'] and not released['retired'] and not released['prefix_protected'] and released['tables']==4 and released['free_pages']+released['prefix_total']==2048 and released['page_indices_unique'] and released['table_slots_unique']
            work_ok &= released['page_indices_sha256']==token_hash(list(range(2048)))
            replay_count=0
            for batch in row['batches']:
                size=len(batch['requests']);eligible=batch['phase']=='decode' and bool(settings['graph_bs']) and size<=max(settings['graph_bs'])
                padded=next(b for b in settings['graph_bs'] if b>=size) if eligible else size
                work_ok &= batch['graph_eligible']==eligible and batch['padded']==padded
                replay_count+=eligible
            work_ok &= row['graph_replays']==replay_count
        checks[f'{variant}_work_and_resources']=bool(work_ok) and len(seen)==len(record['rows'])==16
        reports[variant]=dict(passed=not failures,vectors=len(expected_ids)*64,exact_vectors=exact,
            max_relative_rmse=maximum,failures=failures,evidence_sha256=digest(path))
    journal=json.loads((directory/'processes.json').read_text())
    checks['eight_clean_processes']=len(journal)==8 and all(r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in journal)
    checks['all_12544_vectors_compared']=vectors==12544 and not missing
    integrity=all(checks.values())
    return dict(integrity_passed=integrity,qualification_passed=integrity and all(r['passed'] for r in reports.values()),
        performance_allowed=False,local_single_gpu_complete=False,reference_scope=meta['reference_scope'],
        checks=checks,variants=reports,missing_variants=missing,compared_vectors=vectors,
        original_HF_FP32_equivalence_passed=False,source_sha256=digest(Path(__file__)),
        provenance_sha256=digest(directory/'provenance.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'audit.json'
    if path.exists():raise ValueError('Preserve independent audit')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(integrity_passed=result['integrity_passed'],qualification_passed=result['qualification_passed'],
        variants={k:{n:v[n] for n in ('passed','exact_vectors','max_relative_rmse')} for k,v in result['variants'].items()}),indent=2))
    raise SystemExit(0 if result['qualification_passed'] else 1)
