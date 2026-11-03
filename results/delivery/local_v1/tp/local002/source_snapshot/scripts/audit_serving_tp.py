"""Reopen full-vocabulary TP tensors; local reports cannot authorize TP timing."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import read,sha,write
from serving.tp_preparation import CONFIG,jobs,load_plan


def audit(directory,local=False):
    import torch
    from serving.i2_worker import error_metrics
    c=load_plan();meta=read(directory/'provenance.json')
    if any(sha(ROOT/n)!=s for n,s in meta['source_sha256'].items()):raise ValueError('TP source changed')
    if sha(directory/'workload.json')!=meta['workload_sha256']:raise ValueError('TP workload changed')
    hf=read(directory/'hf_reference.json');assert sha(directory/'hf_reference.pt')==hf['tensor_sha256']
    hf_values=torch.load(directory/'hf_reference.pt',map_location='cpu',weights_only=True)
    selected=jobs(c,'qualify');selected=[j for j in selected if j['world']==1] if local else selected
    results={};evidence={};checks={};loaded={};vectors=0
    for job in selected:
        for rank in range(job['world']):
            p=directory/'raw'/job['label']/f'rank{rank}.json';r=read(p);evidence[str(p.relative_to(directory))]=sha(p)
            key=f'{job["label"]}/rank{rank}'
            checks[key+'/run']=(r['passed'] and r['rank']==rank and r['world']==job['world'] and r['workload_sha256']==meta['workload_sha256'])
            checks[key+'/resources']=all(w['released'] for w in r['waves'])
            workload=read(directory/'workload.json')
            expected={q['request_id']:q for case in workload['cases'] for phase in ('cold','warm') for q in case['waves'][phase]}
            actual={q['request_id']:q for wave in r['waves'] for q in wave['requests']}
            checks[key+'/accounting']=set(expected)==set(actual) and len(r['waves'])==4
            checks[key+'/outputs']=all(q['output_tokens']==len(expected[rid]['continuation_ids']) and
                (q.get('actual_tokens',expected[rid]['continuation_ids'])==expected[rid]['continuation_ids']) for rid,q in actual.items())
            if job['variant'] in ('eager_no_prefix','graph_no_prefix','naive_full'):
                checks[key+'/no_prefix']=all(q['cached_tokens']==0 for q in actual.values())
            else:
                shared=[q for wave in r['waves'] if wave['case_id']=='b3_p512_shared' and wave['phase']=='warm' for q in wave['requests']]
                checks[key+'/warm_prefix_reused']=bool(shared) and all(q['cached_tokens']>=256 for q in shared)
            if 'tensor_file' not in r:
                if not (job['framework']=='nano' and rank>0):raise ValueError('Missing rank logits')
                continue
            tensor=directory/r['tensor_file'];assert sha(tensor)==r['tensor_sha256'];evidence[r['tensor_file']]=sha(tensor)
            values=torch.load(tensor,map_location='cpu',weights_only=True);loaded[key]=values
            if job['framework']=='mini':reference=hf_values
            else:
                base='nano_tp1_eager_no_prefix_qualify_run0/rank0'
                reference=loaded[base]
            errors={rid:error_metrics(value,reference[rid],c['numerics']) for rid,value in values.items()}
            results[key]=errors;vectors+=sum(len(v) for v in errors.values())
            checks[key+'/full_vocabulary']=set(values)==set(expected) and all(v.shape==(8,151936) for v in values.values())
            checks[key+'/numerical']=all(row['passed'] for errors in errors.values() for row in errors)
    evidence.update({'workload.json':sha(directory/'workload.json'),'hf_reference.json':sha(directory/'hf_reference.json'),
        'hf_reference.pt':sha(directory/'hf_reference.pt'),'provenance.json':sha(directory/'provenance.json')})
    return dict(passed=all(checks.values()),local_only=local,multigpu_accepted=not local and all(checks.values()),
        qualified_labels=[j['label'] for j in selected],checks=checks,results=results,vectors=vectors,
        config_sha256=sha(CONFIG),model_manifest_sha256=sha(ROOT/c['model_manifest']),source_sha256=meta['source_sha256'],
        evidence_sha256=evidence,nano_HF_FP32_equivalence_claimed=False,paid_resources_used=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--local',action='store_true')
    args=p.parse_args();target=args.directory/('local_audit.json' if args.local else 'qualification_gate.json')
    if target.exists():raise ValueError('Preserve TP audit')
    result=audit(args.directory,args.local);write(target,result)
    print('TP audit',result['passed'],'vectors',result['vectors'],'multigpu',result['multigpu_accepted'])
    if not result['passed']:raise SystemExit(1)
