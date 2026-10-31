"""Bind independently audited external-reference evidence into a bounded report."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import psutil
    import torch
    from serving.i2_workloads import digest
    from scripts.audit_i2_external_alignment import difference
    torch.set_num_threads(4)
    base = ROOT / 'results/i_minisglang/i2_external_alignment_v1'
    output = base / 'summary.json'
    if output.exists():
        raise ValueError('Preserve report')
    attempts = ('full_naive001', 'full_reproduction001', 'full_radix001', 'mixed_naive001', 'mixed_radix001')
    audits, metas = {}, {}
    for name in attempts:
        directory = base / name
        audit = json.loads((directory / 'audit.json').read_text())
        meta = json.loads((directory / 'provenance.json').read_text())
        if not audit['integrity_passed'] or meta['performance_allowed'] or meta['paid_resources_used']:
            raise ValueError('Evidence not admitted')
        audits[name], metas[name] = audit, meta
        snapshot = directory / 'audit_source_snapshot/scripts/audit_i2_external_alignment.py'
        if digest(snapshot) != audit['auditor_sha256']:
            raise ValueError('Auditor snapshot changed')
        for row in json.loads((directory / 'processes.json').read_text()):
            if psutil.pid_exists(row['pid']):
                raise ValueError('Dedicated process still alive')
    common = metas[attempts[0]]
    for meta in metas.values():
        for key in ('reference', 'model_sha256', 'workload_sha256', 'checkpoint_dtypes', 'series'):
            if meta[key] != common[key]:
                raise ValueError('Cross-attempt binding mismatch: ' + key)
    reproduction = audits['full_reproduction001']
    if not reproduction['checks']['native_observer_reproduces_original'] or not reproduction['checks']['hf_observer_reproduces_frozen_reference']:
        raise ValueError('Original full-cohort results did not reproduce')
    if not reproduction['norm_probe']:
        raise ValueError('Missing independent same-input norm probe')
    if reproduction['summaries']['native_bf16']['numerical_passed']:
        raise ValueError('Unexpected BF16 result; investigate rather than relabel')
    for name in ('full_naive001', 'full_radix001'):
        for mode in ('fp32_math',):
            result = audits[name]['summaries'][mode]
            if result['vectors'] != 1024 or result['passed_vectors'] != 1024 or not result['numerical_passed']:
                raise ValueError('Precision candidate did not pass original complete matrix')
    for name in ('mixed_naive001', 'mixed_radix001'):
        result = audits[name]['summaries']['fp16_residual_fp32']
        if result['vectors'] != 1024 or result['passed_vectors'] != 1024 or not result['numerical_passed']:
            raise ValueError('Mixed precision candidate did not pass complete original matrix')
    controls = json.loads((base / 'audit_negative_controls.json').read_text())
    cpu = json.loads((base / 'cpu_tests_precision_candidate.json').read_text())
    if not controls['passed'] or not cpu['passed'] or cpu['tests'] != 7:
        raise ValueError('Missing CPU or negative controls')
    for name, sha in cpu['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('CPU tested source changed')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    if gpu != '0 MiB, 0 %':
        raise ValueError('GPU has not returned to idle: ' + gpu)
    traces = {}
    for mode, attempt in [('hf_fp32', 'full_reproduction001'), ('native_bf16', 'full_reproduction001'),
                          ('native_fp16', 'full_naive001'), ('fp32_math', 'full_naive001'),
                          ('fp16_residual_fp32', 'mixed_naive001')]:
        record = json.loads((base / attempt / f'{mode}.json').read_text())
        artifact = record['tensors']['trace']
        if digest(ROOT/artifact['path']) != artifact['sha256']:
            raise ValueError('Trace changed')
        traces[mode] = torch.load(ROOT/artifact['path'], weights_only=True, mmap=True)
    layers = {}
    for mode in ('native_bf16', 'native_fp16', 'fp32_math', 'fp16_residual_fp32'):
        if set(traces[mode]) != set(traces['hf_fp32']):
            raise ValueError('Cross-precision operator names differ')
        rows = [dict(name=n, **difference(traces[mode][n], traces['hf_fp32'][n])) for n in traces['native_bf16']]
        if mode in {'native_bf16', 'fp32_math'} and not rows[0]['exact']:
            raise ValueError('BF16 checkpoint values must agree before FP32/BF16 computation')
        if mode in {'native_fp16', 'fp16_residual_fp32'} and (not rows[0]['finite'] or rows[0]['max_abs'] > 2.9802322387695312e-8):
            raise ValueError('Unexpected FP16 embedding conversion error')
        layers[mode + '_vs_hf_fp32'] = dict(first_nonexact=next((r for r in rows if not r['exact']), None), rows=rows)
    layer_path = base / 'cross_precision_layers.json'
    layer_path.write_text(json.dumps(layers, indent=2) + '\n')
    sources = ['scripts/report_i2_external_alignment.py', 'scripts/audit_i2_external_alignment.py']
    result = dict(evidence_class='external_reference_alignment_and_precision_candidate',
        numerical_policy=common['config']['numerics'],
        reference=common['reference'], model_sha256=common['model_sha256'], workload_sha256=common['workload_sha256'],
        attempts={name: dict(variant=audits[name]['variant'], summaries=audits[name]['summaries'],
            independent_audit_sha256=digest(base/name/'audit.json'), audit_checks=audits[name]['check_count']) for name in attempts},
        first_norm_same_input_verified=True, original_full_cohort_outputs_reproduced=True,
        pure_fp16_original_threshold_passed_for_naive_and_radix=all(
            audits[n]['summaries']['native_fp16']['numerical_passed'] for n in ('full_naive001', 'full_radix001')),
        fp16_main_fp32_residual_norm_original_threshold_passed_for_naive_and_radix=True,
        fp32_paged_math_original_threshold_passed_for_naive_and_radix=True,
        original_BF16_HF_FP32_equivalence_passed=False,
        all_seven_precision_variants_qualified=False, new_precision_HTTP_qualified=False,
        natural_generation_equivalence_proven=False, performance_allowed=False,
        production_precision_changed=False, paid_resources_used=False, gpu_end=gpu,
        cpu_tests_sha256=digest(base/'cpu_tests_precision_candidate.json'), audit_controls_sha256=digest(base/'audit_negative_controls.json'),
        layer_comparison_sha256=digest(layer_path), source_sha256={n: digest(ROOT/n) for n in sources})
    output.write_text(json.dumps(result, indent=2) + '\n')
    snapshot = base / 'report_source_snapshot/scripts'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in sources:
        shutil.copy2(ROOT/name, snapshot/Path(name).name)
    print(json.dumps({k: result[k] for k in ('fp16_main_fp32_residual_norm_original_threshold_passed_for_naive_and_radix',
        'original_BF16_HF_FP32_equivalence_passed', 'gpu_end')}, indent=2))


if __name__ == '__main__':
    main()
