"""Bounded rounding diagnostic; cannot unlock the failed v1 timing gate."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-attempt',default='attempt001');args=parser.parse_args()
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from serving.i2_workloads import CONFIG,digest
    from serving.i2_math_reference import BF16MathReference,rms,fused_add_norm
    from serving.i2_worker import error_metrics
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads(CONFIG.read_text());directory=ROOT/config['results_dir']/args.source_attempt
    output=ROOT/config['results_dir']/'rounding_diagnostic.json'
    if output.exists():raise ValueError('Preserve diagnostic')
    target=json.loads((directory/'numeric_naive_full_run0.json').read_text())
    workload=json.loads((directory/'workload.json').read_text())
    model=AutoModelForCausalLM.from_pretrained(ROOT/config['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    reference=BF16MathReference(model)
    rows=[]
    with torch.inference_mode():
        for case in workload['cases']:
            request=case['waves']['cold'][0];rid=request['request_id']
            wave=next(w for w in target['replay']['cases'] if w['case_id']==case['case_id'] and w['phase']=='cold')
            actual=torch.load(ROOT/wave['tensor_file'],mmap=True,weights_only=True)[rid][:4]
            reference.reset();values=[];ids=torch.tensor(request['input_ids'],device='cuda')
            for token in request['continuation_ids'][:4]:
                values.append(reference(ids).detach().cpu().float());ids=torch.tensor([token],device='cuda')
            custom=torch.stack(values)
            hf_values=[];past=None;ids=torch.tensor([request['input_ids']],device='cuda')
            for token in request['continuation_ids'][:4]:
                out=model(input_ids=ids,past_key_values=past,use_cache=True,logits_to_keep=1)
                hf_values.append(out.logits[0,-1].detach().cpu().float());past=out.past_key_values;ids=torch.tensor([[token]],device='cuda')
            rows.append(dict(request_id=rid,native_vs_bf16_hf=error_metrics(actual,torch.stack(hf_values),config['numerics']),
                native_vs_explicit_rounding=error_metrics(actual,custom,config['numerics'])))
            del past,out
    record=dict(stage='I2',evidence_class='rounding_diagnostic_only',performance_allowed=False,
        policy_unchanged=config['numerics'],rows=rows,source_sha256={n:digest(ROOT/n) for n in ('serving/i2_math_reference.py','scripts/diagnose_i2_rounding.py')})
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
