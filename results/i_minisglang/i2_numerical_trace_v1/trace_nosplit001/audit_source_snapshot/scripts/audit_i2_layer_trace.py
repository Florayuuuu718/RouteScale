"""Independently compare full native traces and the frozen unobserved outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def audit(directory):
    import torch
    from serving.i2_attention_reference import metadata_checks,tensor_metrics
    torch.set_num_threads(4)
    meta=json.loads((directory/'provenance.json').read_text())
    config=meta['config'];prefix=config['shared_prefix_tokens']
    workload=json.loads((directory/'workload.json').read_text())
    case=next(c for c in workload['cases'] if c['case_id']=='b1_shared')
    checks=dict(frozen_sources=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items()),
        workload_sha256=digest(directory/'workload.json')==meta['workload_sha256'])
    traces={};provenance={};prefix_rows=[]
    old_root=ROOT/config['results_dir']
    for variant,attempt in (('naive_full','attempt001'),('radix_full','native_diagnostic001')):
        record=json.loads((directory/f'{variant}.json').read_text())
        checks[f'tensors_{variant}']=all(digest(ROOT/v['path'])==v['sha256'] for v in record['tensors'].values())
        traces[variant]=torch.load(ROOT/record['tensors']['trace']['path'],mmap=True,weights_only=True)
        logits=torch.load(ROOT/record['tensors']['logits']['path'],mmap=True,weights_only=True)
        comparison_run=meta.get('comparison_run')
        old_path=ROOT/comparison_run/f'{variant}.json' if comparison_run else old_root/attempt/f'numeric_{variant}_run0.json'
        old=json.loads(old_path.read_text())
        provenance[variant]=digest(old_path)
        for phase in ('cold','warm'):
            if comparison_run:
                path=ROOT/old['tensor']['path'];sha=old['tensor']['sha256']
            else:
                wave=next(r for r in old['replay']['cases'] if r['case_id']=='b1_shared' and r['phase']==phase)
                path=ROOT/wave['tensor_file'];sha=wave['tensor_sha256']
            checks[f'original_hash_{variant}_{phase}']=digest(path)==sha
            expected=torch.load(path,mmap=True,weights_only=True)
            rid=case['waves'][phase][0]['request_id']
            checks[f'trace_did_not_change_64_logits_{variant}_{phase}']=torch.equal(logits[rid],expected[rid]) and logits[rid].shape==(64,151936)
            cached=prefix if phase=='warm' and variant=='radix_full' else 0
            flags=metadata_checks(traces[variant]['metadata'][phase],case['waves'][phase][0]['input_ids'],cached,config['engine']['num_pages'])
            checks.update({f'{variant}_{phase}_{k}':v for k,v in flags.items()})
            released=next(r['released'] for r in record['rows'] if r['phase']==phase)
            checks[f'resources_{variant}_{phase}']=not released['pending'] and not released['running'] and not released['retired'] and released['table_slots_unique'] and released['page_indices_unique'] and released['prefix_protected']==0
    naive,radix=traces['naive_full'],traces['radix_full']
    checks['same_operator_order']=naive['order']==radix['order']
    checks['complete_28_layers']=all(f'{i}.attention' in naive['ops'] for i in range(28))
    operator_rows=[dict(operation=n,**tensor_metrics(radix['ops'][n],naive['ops'][n])) for n in naive['order']]
    checks['finite_operators']=all(r['finite'] for r in operator_rows)
    for i in range(28):
        for kind in ('k','v'):
            name=f'{i}.{kind}'
            pairs={
                'cold_naive_vs_radix':(radix['cold_kv'][name],naive['cold_kv'][name]),
                'naive_recomputed_prefix':(naive['warm_kv'][name][:prefix],naive['cold_kv'][name]),
                'radix_prefix_before':(radix['prefix_before'][name],radix['cold_kv'][name]),
                'radix_prefix_after':(radix['warm_kv'][name][:prefix],radix['prefix_before'][name]),
            }
            for label,(a,b) in pairs.items():
                row=dict(layer=i,kind=kind,comparison=label,**tensor_metrics(a,b))
                prefix_rows.append(row)
    checks['all_prefix_values_exact']=all(r['exact'] for r in prefix_rows)
    processes=json.loads((directory/'processes.json').read_text())
    checks['two_clean_processes']=len(processes)==2 and all(r['returncode']==0 and r['gpu_after']=='0 MiB' and digest(ROOT/r['log'])==r['log_sha256'] for r in processes)
    return dict(evidence_class='first_native_divergence_diagnostic',performance_allowed=False,
        localization_valid=all(checks.values()),checks=checks,
        first_exact_difference=next((r for r in operator_rows if not r['exact']),None),
        operators=operator_rows,prefix_checks=prefix_rows,original_evidence_sha256=provenance,
        trace_evidence_sha256={v:digest(directory/f'{v}.json') for v in traces},
        source_sha256={n:digest(ROOT/n) for n in ('scripts/audit_i2_layer_trace.py','serving/i2_attention_reference.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'audit.json'
    if args.check_only:
        if json.loads(path.read_text())!=result:raise ValueError('Trace audit changed')
    elif path.exists():raise ValueError('Preserve audit')
    else:path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('localization_valid','first_exact_difference')},indent=2))
    raise SystemExit(0 if result['localization_valid'] else 1)
