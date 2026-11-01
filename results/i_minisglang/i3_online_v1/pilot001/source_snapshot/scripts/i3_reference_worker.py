"""Fresh HF FP32 reference along each actual native HTTP generation prefix."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def main():
    import torch
    from transformers import AutoModelForCausalLM
    from scripts.run_i2_precision import precision_setup
    from serving.i2_worker import error_metrics
    precision_setup();torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--variant',required=True)
    args=p.parse_args();meta=read(args.directory/'provenance.json');config=meta['config'];workload=read(args.directory/'workload.json')
    service=args.directory/f'qualification_{args.variant}';run=read(service/'run.json');events=read(service/'events.json')
    assignments={e['external_id']:e['uid'] for e in events if e['event']=='frontend_new_user'}
    native_dir=ROOT/run['tensors_dir'];target_dir=native_dir.parent/f'hf_{args.variant}'
    target_dir.mkdir(parents=True,exist_ok=False)
    model=AutoModelForCausalLM.from_pretrained(ROOT/config['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='eager').float().cuda().eval()
    rows={}
    with torch.inference_mode():
        for request in workload['warmup']+workload['requests']:
            rid=request['request_id'];uid=assignments[rid]
            actual_path=native_dir/f'uid_{uid}.pt'
            actual=torch.load(actual_path,weights_only=True)
            tokens=[e['token_id'] for e in events if e['event']=='scheduler_token' and e['uid']==uid]
            assert actual['input_ids']==request['input_ids'] and len(tokens)==request['output_budget']
            inputs=torch.tensor([actual['input_ids']],device='cuda');past=None;values=[]
            for token in tokens:
                out=model(input_ids=inputs,past_key_values=past,use_cache=True,logits_to_keep=1)
                values.append(out.logits[0,-1].detach().cpu().contiguous());past=out.past_key_values
                inputs=torch.tensor([[token]],device='cuda')
            value=torch.stack(values);path=target_dir/f'uid_{uid}.pt';torch.save(value,path)
            errors=error_metrics(actual['logits'],value,config['numerics'])
            rows[rid]=dict(uid=uid,actual_path=str(actual_path.relative_to(ROOT)),actual_sha256=digest(actual_path),
                reference_path=str(path.relative_to(ROOT)),reference_sha256=digest(path),tokens=tokens,errors=errors)
            print(args.variant,rid,'vectors',len(values),'passed',all(e['passed'] for e in errors),flush=True)
            del past,out
    passed=all(e['passed'] for r in rows.values() for e in r['errors'])
    write(args.directory/f'hf_{args.variant}.json',dict(passed=passed,requests=rows,
        reference='same BF16 checkpoint promoted to HF FP32 eager; actual native greedy continuation is teacher forced',
        thresholds=config['numerics'],vectors=sum(len(r['tokens']) for r in rows.values())))
    return 0  # Preserve every comparison, including numerical failures, before auditing.


if __name__=='__main__':raise SystemExit(main())
