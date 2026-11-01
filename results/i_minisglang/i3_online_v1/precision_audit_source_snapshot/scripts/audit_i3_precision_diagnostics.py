"""Read-only independent tensor audit of failed-history precision probes."""
from pathlib import Path
import math
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def main():
    import torch
    torch.set_num_threads(4)
    from serving.i2_worker import error_metrics
    base=ROOT/'results/i_minisglang/i3_online_v1';original=read(base/'qual001/hf_radix.json')['requests']
    config=read(base/'qual001/provenance.json')['config'];checks={};profiles={}
    for upper in (-1,0,8,16):
        path=base/f'trace_upper{upper}';report=read(path/'probe.json')
        checks[f'{upper}_source_snapshots']=all(digest(path/'source_snapshot'/n)==sha for n,sha in report['source_sha256'].items())
        checks[f'{upper}_tensor_hash']=digest(ROOT/report['tensor']['path'])==report['tensor']['sha256']
        checks[f'{upper}_reference_binding']=digest(base/'qual001/hf_radix.json')==report['original_reference_sha256']
        data=torch.load(ROOT/report['tensor']['path'],weights_only=True,mmap=True)
        checks[f'{upper}_coverage']=set(data)==set(original) and report['checked_batches']==report['recorded_batches']==975
        total=[];exact=0;consistent=True
        for rid,value in data.items():
            record=original[rid]
            reference=torch.load(ROOT/record['reference_path'],weights_only=True)
            native=torch.load(ROOT/record['actual_path'],weights_only=True)['logits']
            errors=error_metrics(value,reference,config['numerics']);total+=errors
            exact+=sum(torch.equal(a,b) for a,b in zip(value,native,strict=True))
            old=report['errors'][rid]
            consistent &= len(errors)==len(old) and all(
                all(a[k]==b[k] for k in ('finite','passed','top1_equal','tolerance_violations'))
                and all(math.isclose(a[k],b[k],rel_tol=1e-6,abs_tol=1e-9) for k in ('max_abs','relative_rmse'))
                for a,b in zip(errors,old,strict=True))
        failed=sum(not e['passed'] for e in total);maximum=max(e['relative_rmse'] for e in total)
        checks[f'{upper}_recomputed_metrics']=bool(consistent) and failed==report['failed_vectors'] and len(total)==report['vectors']==1664 and exact==report['exact_original_vectors'] and math.isclose(maximum,report['max_relative_rmse'],rel_tol=1e-6)
        profiles[str(upper)]=dict(vectors=len(total),failed_vectors=failed,max_relative_rmse=maximum,exact_original_vectors=exact)
        print('Independent precision probe',upper,profiles[str(upper)],flush=True)
    checks['original_failure_exactly_reproduced']=profiles['-1']['failed_vectors']==2 and profiles['-1']['exact_original_vectors']==1664
    checks['head_only_failure_preserved']=profiles['0']['failed_vectors']>0
    checks['selected_upper16_all_pass']=profiles['16']['failed_vectors']==0
    result=dict(integrity_passed=all(checks.values()),checks=checks,profiles=profiles,
        diagnostic_only=True,performance_allowed=False,source_sha256=digest(Path(__file__)))
    output=base/'precision_bundle_audit.json'
    if output.exists():raise ValueError('Preserve bundle audit')
    write(output,result);dest=base/'precision_audit_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),dest/Path(__file__).name)
    if not result['integrity_passed']:raise ValueError('Diagnostic integrity failed')


if __name__=='__main__':main()
