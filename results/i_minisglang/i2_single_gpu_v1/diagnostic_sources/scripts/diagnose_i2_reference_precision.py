"""Check the v1 gate's precision boundary within Transformers itself."""
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM
    from serving.i2_workloads import CONFIG,digest
    from serving.i2_worker import error_metrics
    config=json.loads(CONFIG.read_text())
    output=ROOT/config['results_dir']/'reference_precision_diagnostic.json'
    if output.exists():raise ValueError('Preserve diagnostic')
    workload=json.loads((ROOT/config['workload_path']).read_text())
    reference_path=ROOT/config['validation_dir']/'attempt001/reference.pt'
    fp32=torch.load(reference_path,mmap=True,weights_only=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(ROOT/config['model_path'],local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    rows=[]
    with torch.inference_mode():
        for case in workload['cases']:
            request=case['waves']['cold'][0];rid=request['request_id']
            values=[];past=None;ids=torch.tensor([request['input_ids']],device='cuda')
            for token in request['continuation_ids'][:4]:
                out=model(input_ids=ids,past_key_values=past,use_cache=True,logits_to_keep=1)
                values.append(out.logits[0,-1].detach().cpu().float());past=out.past_key_values
                ids=torch.tensor([[token]],device='cuda')
            rows.append(dict(request_id=rid,errors=error_metrics(torch.stack(values),fp32[rid][:4],config['numerics'])))
            del out,past
    output.write_text(json.dumps(dict(evidence_class='reference_precision_diagnostic_only',
        performance_allowed=False,policy_unchanged=config['numerics'],rows=rows,
        source_sha256=digest(Path(__file__)),reference_sha256=digest(reference_path)),indent=2)+'\n')
    for row in rows:
        print(row['request_id'],'passed',sum(e['passed'] for e in row['errors']),'/ 4',
            'max_relative_rmse',max(e['relative_rmse'] for e in row['errors']))


if __name__=='__main__':main()
