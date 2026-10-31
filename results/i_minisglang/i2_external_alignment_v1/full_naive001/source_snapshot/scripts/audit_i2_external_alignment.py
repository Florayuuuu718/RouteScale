"""Reopen I2 external-alignment tensors and recompute numerical conclusions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def difference(actual, expected):
    import torch
    a, b = actual.float(), expected.float()
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {a.shape}, {b.shape}')
    delta = a - b
    return dict(exact=bool(torch.equal(a, b)), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        max_abs=float(delta.abs().max()), relative_l2=float(delta.norm() / b.norm().clamp_min(1e-12)),
        unequal_elements=int((a != b).sum()), shape=list(a.shape))


def audit(directory):
    import torch
    from serving.i2_workloads import digest
    torch.set_num_threads(4)
    meta = json.loads((directory / 'provenance.json').read_text())
    checks = {}

    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise ValueError('Audit failed: ' + name)

    for name, sha in meta['source_sha256'].items():
        check('snapshot:' + name, digest(directory / 'source_snapshot' / name) == sha)
    check('hf_snapshot', digest(directory / 'source_snapshot/modeling_qwen3.py') == meta['hf_source_sha256'])
    for name, sha in meta['model_sha256'].items():
        check('model:' + name, digest(ROOT / meta['config']['model_path'] / name) == sha)
    check('workload', digest(directory / 'workload.json') == meta['workload_sha256'])
    check('reference', digest(ROOT / meta['reference']['path']) == meta['reference']['sha256'])
    old = json.loads((ROOT / 'results/i_minisglang/i2_single_gpu_v1/attempt001/reference.json').read_text())
    check('original_reference_binding', old['reference']['tensor_sha256'] == meta['reference']['sha256'])
    policy = meta['config']['numerics']
    check('unchanged_threshold', policy['atol'] == .125 and policy['rtol'] == .02 and policy['max_relative_rmse'] == .005)
    check('bf16_checkpoint', meta['checkpoint_dtypes'] == ['BF16'])
    check('no_performance', meta['performance_allowed'] is False)
    frozen = torch.load(ROOT / meta['reference']['path'], mmap=True, weights_only=True)
    workload = json.loads((directory / 'workload.json').read_text())
    expected_requests = {r['request_id']: r for c in workload['cases'] for phase in
        (('cold',) if meta['scope'] == 'pilot' else ('cold', 'warm')) for r in
        (c['waves'][phase][:1] if meta['scope'] == 'pilot' else c['waves'][phase])}
    journal = json.loads((directory / 'processes.json').read_text())
    check('process_set', {r['label'] for r in journal} == set(meta['modes']) and len(journal) == len(meta['modes']))
    for row in journal:
        check('process:' + row['label'], row['status'] == 'pass' and row['returncode'] == 0)
        check('log:' + row['label'], digest(ROOT / row['log']) == row['log_sha256'])
    records, tensors, traces, summaries = {}, {}, {}, {}
    for mode in meta['modes']:
        r = json.loads((directory / f'{mode}.json').read_text())
        records[mode] = r
        for kind, artifact in r['tensors'].items():
            check(mode + ':' + kind + ':sha256', digest(ROOT / artifact['path']) == artifact['sha256'])
        values = torch.load(ROOT / r['tensors']['logits']['path'], mmap=True, weights_only=True)
        tensors[mode] = values
        traces[mode] = torch.load(ROOT / r['tensors']['trace']['path'], mmap=True, weights_only=True)
        check(mode + ':requests', set(values) == set(expected_requests) == {row['request_id'] for row in r['rows']})
        rowmap = {row['request_id']: row for row in r['rows']}
        errors = []
        for rid, actual in values.items():
            count = 4 if meta['scope'] == 'pilot' else 64
            check(mode + ':' + rid + ':shape', list(actual.shape) == [count, 151936])
            logged = rowmap[rid]['errors']
            check(mode + ':' + rid + ':error_count', len(logged) == count)
            for i in range(count):
                a, b = actual[i].float(), frozen[rid][i].float()
                d = (a-b).abs()
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                rel = float((a-b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12))
                violations = int((d > policy['atol'] + policy['rtol'] * b.abs()).sum())
                passed = finite and violations == 0 and rel <= policy['max_relative_rmse']
                item = dict(finite=finite, relative_rmse=rel, max_abs=float(d.max()), tolerance_violations=violations,
                    passed=passed, top1_equal=int(a.argmax()) == int(b.argmax()))
                check(f'{mode}:{rid}:{i}:metrics', all(
                    math.isclose(logged[i][k], v, rel_tol=1e-6, abs_tol=1e-9) if isinstance(v, float)
                    else logged[i][k] == v for k, v in item.items()))
                errors.append(item)
            if not mode.startswith('hf_'):
                released = rowmap[rid]['released']
                check(mode + ':' + rid + ':released', released['table_slots_unique'] and released['page_indices_unique'] and
                    not released['pending'] and not released['running'] and not released['retired'] and
                    released['prefix_protected'] == 0)
        summary = dict(vectors=len(errors), passed_vectors=sum(e['passed'] for e in errors),
            max_relative_rmse=max(e['relative_rmse'] for e in errors), max_abs=max(e['max_abs'] for e in errors),
            tolerance_violations=sum(e['tolerance_violations'] for e in errors),
            top1_equal_vectors=sum(e['top1_equal'] for e in errors), numerical_passed=all(e['passed'] for e in errors))
        for key in ('vectors', 'passed_vectors', 'max_relative_rmse', 'max_abs', 'numerical_passed'):
            check(mode + ':summary:' + key, summary[key] == r[key])
        summaries[mode] = summary
    if 'hf_fp32' in tensors:
        check('hf_observer_reproduces_frozen_reference', all(torch.equal(v, frozen[rid]) if meta['scope'] == 'full'
            else torch.equal(v, frozen[rid][:4]) for rid, v in tensors['hf_fp32'].items()))
    if 'native_bf16' in tensors and meta['variant'] == 'naive_full':
        native = json.loads((ROOT / 'results/i_minisglang/i2_single_gpu_v1/attempt001/numeric_naive_full_run0.json').read_text())
        original = {}
        for wave in native['replay']['cases']:
            original.update(torch.load(ROOT / wave['tensor_file'], mmap=True, weights_only=True))
        check('native_observer_reproduces_original', all(torch.equal(v, original[rid][:len(v)]) for rid, v in tensors['native_bf16'].items()))
    comparisons = {}
    for actual, reference in [('native_bf16', 'hf_fp32'), ('native_bf16', 'hf_bf16'), ('hf_bf16', 'hf_fp32'),
                              ('residual_fp32', 'hf_fp32'), ('native_fp16', 'hf_fp32'), ('fp32_math', 'hf_fp32')]:
        if actual not in traces or reference not in traces:
            continue
        a, b = traces[actual], traces[reference]
        check(actual + '_vs_' + reference + ':trace_set', set(a) == set(b) and bool(a))
        # Native insertion order follows execution; HF norm pre-hooks have reversed tuple order.
        order = list(traces.get('native_bf16', a))
        rows = [dict(name=n, **difference(a[n], b[n])) for n in order]
        first = next((r for r in rows if not r['exact']), None)
        comparisons[actual + '_vs_' + reference] = dict(first_nonexact=first, rows=rows)
    norm_probe = None
    if (directory / 'norm_probe.json').exists():
        probe = json.loads((directory / 'norm_probe.json').read_text())
        for name, sha in probe['source_sha256'].items():
            check('norm_snapshot:' + name, digest(directory / 'norm_probe_source_snapshot' / name) == sha)
        artifact = probe['tensor']
        check('norm_tensor_hash', digest(ROOT / artifact['path']) == artifact['sha256'])
        values = torch.load(ROOT / artifact['path'], weights_only=True, mmap=True)
        check('norm_same_input', all(torch.equal(values['input'].float(), traces[n]['embedding'].float())
            for n in ('native_bf16', 'hf_bf16', 'hf_fp32')))
        pairs = dict(flashinfer_reproduces_native_trace=(values['flashinfer'], traces['native_bf16']['0.input_layernorm.0']),
            hf_bf16_reproduces_hf_trace=(values['hf_bf16'], traces['hf_bf16']['0.input_layernorm.0']),
            fp32_reproduces_hf_trace=(values['fp32'], traces['hf_fp32']['0.input_layernorm.0']),
            native_vs_hf_bf16=(values['flashinfer'], values['hf_bf16']),
            native_vs_fp32=(values['flashinfer'], values['fp32']),
            native_vs_single_rounding_math=(values['flashinfer'], values['fused_math']))
        norm_probe = {name: difference(*pair) for name, pair in pairs.items()}
        check('norm_comparisons', norm_probe == probe['comparisons'])
        check('norm_reproduced', probe['passed'] and all(norm_probe[n]['exact'] for n in
            ('flashinfer_reproduces_native_trace', 'hf_bf16_reproduces_hf_trace', 'fp32_reproduces_hf_trace')))
        x = values['input'].float()
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * values['weight'].float()
        check('norm_independent_cpu_formula', torch.allclose(expected, values['fp32'], atol=2e-6, rtol=1e-6))
    return dict(stage='I2', integrity_passed=all(checks.values()), check_count=len(checks), checks=checks,
        scope=meta['scope'], variant=meta['variant'], summaries=summaries, comparisons=comparisons,
        norm_probe=norm_probe,
        original_BF16_gate_repaired=False, performance_allowed=False,
        auditor_sha256=digest(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / 'audit.json'
    if path.exists():
        raise ValueError('Preserve audit; call audit() directly to recheck')
    result = audit(args.directory)
    path.write_text(json.dumps(result, indent=2) + '\n')
    import shutil
    snapshot = args.directory / 'audit_source_snapshot' / 'scripts'
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    print(json.dumps({k: result[k] for k in ('integrity_passed', 'check_count', 'summaries')}, indent=2))


if __name__ == '__main__':
    main()
