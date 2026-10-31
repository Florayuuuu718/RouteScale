"""Independent CPU FP64 and full-tensor audit of the planning intervention."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def audit(directory):
    import torch
    from serving.i2_attention_reference import tensor_metrics
    torch.set_num_threads(4)
    meta=json.loads((directory/'provenance.json').read_text())
    stored=json.loads((directory/'result.json').read_text())
    checks=dict(sources=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items()),
        input_hashes=all(digest(ROOT/v['path'])==v['sha256'] for v in meta['inputs'].values()),
        library_source_hashes=all(digest(ROOT/n)==sha for n,sha in meta['library_source_sha256'].items()),
        output_hash=digest(ROOT/stored['output']['path'])==stored['output']['sha256'])
    output=torch.load(ROOT/stored['output']['path'],mmap=True,weights_only=True)
    trace=torch.load(ROOT/meta['inputs']['naive_full']['path'],mmap=True,weights_only=True)
    q,k,v=[trace['warm_kv'][n].double() for n in ('0.q_all','0.k','0.v')]
    q=q[-128:];groups=q.shape[1]//k.shape[1]
    k=k.repeat_interleave(groups,dim=1);v=v.repeat_interleave(groups,dim=1)
    score=torch.einsum('thd,shd->hts',q,k)/math.sqrt(q.shape[-1])
    mask=torch.arange(len(k))[None,:] <= torch.arange(384,512)[:,None]
    score.masked_fill_(~mask[None],float('-inf'))
    expected=torch.einsum('hts,shd->thd',score.softmax(-1),v)
    fp64_delta=float((expected-output['math64']).abs().max())
    checks['independent_CPU_FP64_math']=fp64_delta<1e-10
    checks['math32_vs_math64']=tensor_metrics(output['math32'],output['math64'])==stored['math32_vs_math64']
    for variant,record in meta['inputs'].items():
        actual_trace=torch.load(ROOT/record['path'],mmap=True,weights_only=True)
        actual=tensor_metrics(output[f'auto.{variant}'],actual_trace['ops']['0.attention'])
        checks[f'reproduce_original_{variant}']=actual==stored['original_trace_reproduced'][variant] and actual['exact']
    for row in stored['rows']:
        policy=row['policy'];full=output[f'{policy}.naive_full'];suffix=output[f'{policy}.radix_full']
        for key,a,b in (('full_vs_suffix',suffix,full),('full_vs_math32',full,output['math32']),('suffix_vs_math32',suffix,output['math32'])):
            observed=tensor_metrics(a,b);reported=row[key]
            # CUDA and CPU reductions can differ slightly. Counts and equality
            # are exact; scalar relative norms are compared at rounding scale.
            flags=[observed[k]==reported[k] for k in ('finite','exact','max_abs','different_elements')]
            flags.append(math.isclose(observed['relative_l2'],reported['relative_l2'],abs_tol=1e-8,rel_tol=1e-5))
            checks[f'{policy}_{key}']=all(flags)
        checks[f'{policy}_plan_metadata']=all(p['cta_tile_q']==p['vector'][3] and p['split_kv']==bool(p['vector'][14]) for p in row['plans'].values())
    by_policy={r['policy']:r for r in stored['rows']}
    checks['auto_changes_split']=not by_policy['auto']['plans']['naive_full']['split_kv'] and by_policy['auto']['plans']['radix_full']['split_kv'] and not by_policy['auto']['full_vs_suffix']['exact']
    checks['no_split_removes_local_difference']=by_policy['no_split']['full_vs_suffix']['exact'] and all(not p['split_kv'] for p in by_policy['no_split']['plans'].values())
    checks['common_fixed128_removes_local_difference']=by_policy['fixed128']['full_vs_suffix']['exact'] and all(p['kv_chunk_size']==128 for p in by_policy['fixed128']['plans'].values())
    journal=json.loads((directory/'processes.json').read_text())
    checks['clean_process']=len(journal)==1 and journal[0]['returncode']==0 and journal[0]['gpu_after']=='0 MiB' and digest(ROOT/journal[0]['log'])==journal[0]['log_sha256']
    return dict(integrity_passed=all(checks.values()),checks=checks,CPU_FP64_max_abs=fp64_delta,
        performance_allowed=False,scope='layer0_same_QKV_planning_only',
        source_sha256={n:digest(ROOT/n) for n in ('scripts/audit_i2_attention_probe.py','serving/i2_attention_reference.py')},
        result_sha256=digest(directory/'result.json'),provenance_sha256=digest(directory/'provenance.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory);path=args.directory/'audit.json'
    if path.exists():raise ValueError('Preserve audit')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
