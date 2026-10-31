"""Separate native strategy drift from the retained external-reference failure."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import CONFIG,digest
from serving.i2_worker import error_metrics


def main():
    import torch
    torch.set_num_threads(4)
    config=json.loads(CONFIG.read_text());base=ROOT/config['results_dir']
    left=base/'attempt001/numeric_naive_full_run0.json'
    right=base/'native_diagnostic001/numeric_radix_full_run0.json'
    a=json.loads(left.read_text());b=json.loads(right.read_text())
    for name in ('config_sha256','workload_sha256','patched_tree','packages','source_sha256','precision'):
        if a[name]!=b[name]:raise ValueError(f'Native diagnostic changed {name}')
    rows=[]
    for wave in a['replay']['cases']:
        other=next(w for w in b['replay']['cases'] if w['case_id']==wave['case_id'] and w['phase']==wave['phase'])
        expected=torch.load(ROOT/wave['tensor_file'],mmap=True,weights_only=True)
        actual=torch.load(ROOT/other['tensor_file'],mmap=True,weights_only=True)
        for rid in expected:
            metrics=error_metrics(actual[rid],expected[rid],config['numerics'])
            rows.append(dict(request_id=rid,vectors=len(metrics),passed=all(m['passed'] for m in metrics),
                max_abs=max(m['max_abs'] for m in metrics),max_relative_rmse=max(m['relative_rmse'] for m in metrics),
                violations=sum(m['tolerance_violations'] for m in metrics),exact=bool(torch.equal(actual[rid],expected[rid]))))
    out=base/'native_strategy_diagnostic.json'
    if out.exists():raise ValueError('Preserve diagnostic')
    result=dict(evidence_class='diagnostic_not_performance_gate',performance_allowed=False,
        compared_vectors=sum(r['vectors'] for r in rows),passed=all(r['passed'] for r in rows),rows=rows,
        baseline_sha256=digest(left),candidate_sha256=digest(right),policy=config['numerics'],source_sha256=digest(__file__))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
