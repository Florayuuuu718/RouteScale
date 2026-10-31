"""Re-open the causal diagnostic tensors; preserve the original failed scope."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def compatible(a,b):
    return all(a[k]==b[k] for k in ('finite','exact','max_abs','different_elements')) and math.isclose(a['relative_l2'],b['relative_l2'],abs_tol=1e-8,rel_tol=1e-5)


def audit():
    import torch
    from serving.i2_attention_reference import tensor_metrics
    from scripts.audit_i2_layer_trace import audit as trace_audit
    from scripts.audit_i2_attention_probe import audit as attention_audit
    torch.set_num_threads(4)
    base=ROOT/'results/i_minisglang/i2_numerical_trace_v1';checks={}
    for name in ('trace001','trace_nosplit001'):
        directory=base/name;old=json.loads((directory/'audit.json').read_text());new=trace_audit(directory)
        for n,sha in old['source_sha256'].items():
            path=directory/'audit_source_snapshot'/n
            checks[f'{name}_auditor_{n}']=digest(path if path.exists() else ROOT/n)==sha
        old.pop('source_sha256');new.pop('source_sha256')
        checks[f'{name}_independent_recompute']=old==new and new['localization_valid']
    directory=base/'trace001/probe001'
    checks['same_QKV_attention_probe']=attention_audit(directory)==json.loads((directory/'audit.json').read_text())
    directory=base/'trace_nosplit001/downproj001';meta=json.loads((directory/'provenance.json').read_text());record=json.loads((directory/'result.json').read_text())
    checks['down_projection_sources']=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items())
    checks['down_projection_inputs']=all(digest(ROOT/v['path'])==v['sha256'] for v in meta['inputs'].values())
    checks['down_projection_output_hash']=digest(ROOT/record['output']['path'])==record['output']['sha256']
    output=torch.load(ROOT/record['output']['path'],mmap=True,weights_only=True)
    traces={v:torch.load(ROOT/r['path'],mmap=True,weights_only=True) for v,r in meta['inputs'].items()}
    pairs={'full_reproduces_trace':(output['full512'],traces['naive_full']['ops']['0.down_proj']),
        'short_reproduces_trace':(output['short128'],traces['radix_full']['ops']['0.down_proj']),
        'full_vs_short':(output['short128'],output['full512']),
        'full_vs_controlled':(output['controlled512'],output['full512']),
        'full_vs_math32':(output['full512'],output['math32']),
        'short_vs_math32':(output['short128'],output['math32'])}
    for name,(a,b) in pairs.items():checks[f'down_projection_{name}']=compatible(tensor_metrics(a,b),record[name])
    checks['down_projection_input_exact']=torch.equal(traces['naive_full']['ops']['0.silu_mul'],traces['radix_full']['ops']['0.silu_mul'])
    checks['down_projection_geometry_isolated']=record['full_reproduces_trace']['exact'] and record['short_reproduces_trace']['exact'] and not record['full_vs_short']['exact'] and record['full_vs_controlled']['exact']
    for name in ('cache_nosplit001','cache_shape001'):
        directory=base/name;meta=json.loads((directory/'provenance.json').read_text());comparison=json.loads((directory/'comparison.json').read_text())
        arrays={}
        for variant in ('naive_full','radix_full'):
            r=json.loads((directory/f'{variant}.json').read_text());checks[f'{name}_{variant}_hash']=digest(ROOT/r['tensor']['path'])==r['tensor']['sha256']
            arrays[variant]=torch.load(ROOT/r['tensor']['path'],mmap=True,weights_only=True)
        for row in comparison['rows']:
            rid=row['request_id'];checks[f'{name}_{rid}_equality']=torch.equal(arrays['naive_full'][rid],arrays['radix_full'][rid])==row['exact']
        checks[f'{name}_no_performance_gate']=comparison['performance_allowed'] is False
        checks[f'{name}_source_snapshots']=all(digest(directory/'source_snapshot'/n)==sha for n,sha in meta['source_sha256'].items())
    # Documentation can add links to later evidence; all frozen v1 measurements,
    # configs and source snapshots must remain byte-for-byte at the prior commit.
    result=subprocess.run(['git','diff','--quiet','11f3b29','--','results/i_minisglang/i2_single_gpu_v1',
        ':(exclude)results/i_minisglang/i2_single_gpu_v1/README.md','config/serving/i2_single_gpu.json',
        'patches/minisgl-i1','patches/minisgl-i2','results/i_minisglang/i1_repaired'],cwd=ROOT)
    checks['original_frozen_evidence_unchanged']=result.returncode==0
    cpu_dir=ROOT/'results/i_minisglang/i2_shape_qualification_v1';cpu=json.loads((cpu_dir/'cpu_tests.json').read_text())
    checks['twelve_new_CPU_tests']=cpu['new_tests_passed']==12 and all(digest(ROOT/n)==sha for n,sha in cpu['source_sha256'].items()) and all(r['returncode']==0 and digest(cpu_dir/r['log'])==r['sha256'] for r in cpu['runs'])
    return dict(diagnostic_integrity_passed=all(checks.values()),checks=checks,performance_allowed=False,
        scope='two_successive_prefill_divergences_and_common_policy',original_HF_FP32_equivalence_passed=False,
        source_sha256=digest(Path(__file__)))


if __name__=='__main__':
    result=audit();path=ROOT/'results/i_minisglang/i2_numerical_trace_v1/independent_bundle_audit.json'
    if path.exists():raise ValueError('Preserve diagnostic bundle audit')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['diagnostic_integrity_passed'] else 1)
