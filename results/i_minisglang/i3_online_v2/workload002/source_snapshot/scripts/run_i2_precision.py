"""Serial I2 precision qualification, real HTTP verification and gated timing."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.i2_workloads import digest, token_hash

SPEC = ROOT / 'config/serving/i2_precision.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load_config():
    spec = read(SPEC)
    return dict(read(ROOT/spec['base_config']), **spec)


def precision_setup():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261031)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def check_sources(meta):
    for name, sha in meta['source_sha256'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Frozen source changed: ' + name)


def qualification_worker(directory, mode, variant, kind):
    import torch
    from serving.i2_worker import error_metrics, telemetry
    meta = read(directory/'provenance.json')
    check_sources(meta)
    precision_setup()
    config = meta['config']
    suites = read(directory/'workload.json')['suites']
    validation = ROOT/meta['validation_dir']
    label = kind if kind in {'hf_reference', 'native_reference'} else f'{mode}_{variant}'
    path = validation/f'{label}.pt'
    if (directory/f'{label}.json').exists() or path.exists():
        raise ValueError('Preserve numerical result')
    tensors, rows, errors, observed = {}, [], {}, {}
    reference = None
    if kind == 'numeric':
        if meta['stage'] == 'smoke':
            reference = torch.load(ROOT/meta['original_reference']['path'], weights_only=True, mmap=True)
        else:
            ref = 'hf_reference' if mode == 'mixed' else 'native_reference'
            reference = torch.load(validation/f'{ref}.pt', weights_only=True, mmap=True)
    scheduler = None
    with torch.inference_mode():
        try:
            if kind == 'hf_reference':
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(ROOT/config['model_path'], local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='eager').float().cuda().eval()
                observed = dict(reference='BF16 checkpoint promoted to FP32 HF eager')
            else:
                sys.path.insert(0, str(ROOT/'runs/vendor/mini-sglang-i2/python'))
                from serving.i2_precision_runtime import PrecisionScheduler
                from serving.i2_runtime import flush_cache
                scheduler = PrecisionScheduler(config, variant, capture_logits=True, mode=mode)
                observed = scheduler.engine._i2_precision_observed
            for suite in suites:
                for case in suite['workload']['cases']:
                    if scheduler is not None:
                        flush_cache(scheduler, poison=True)
                    for phase in ('cold', 'warm'):
                        requests = case['waves'][phase]
                        if meta['stage'] == 'smoke':
                            requests = [dict(r, continuation_ids=r['continuation_ids'][:4]) for r in requests]
                        if scheduler is None:
                            values = {}
                            for request in requests:
                                inputs = torch.tensor([request['input_ids']], device='cuda')
                                past, logits = None, []
                                for forced in request['continuation_ids']:
                                    out = model(input_ids=inputs, past_key_values=past, use_cache=True, logits_to_keep=1)
                                    logits.append(out.logits[0, -1].detach().cpu().contiguous())
                                    past = out.past_key_values
                                    inputs = torch.tensor([[forced]], device='cuda')
                                values[request['request_id']] = torch.stack(logits)
                                del past, out
                        else:
                            row, values, batches = scheduler.wave(requests)
                            expected_hit = suite['config']['shared_prefix_tokens'] if phase == 'warm' and case['prefix_mode'] == 'shared' and config['variants'][variant]['cache'] == 'radix' else 0
                            if phase == 'warm' and any(r['cached_tokens'] != expected_hit for r in row['requests']):
                                raise ValueError('Warm prefix hit mismatch')
                            row.update(suite=suite['name'], case_id=case['case_id'], phase=phase, batches=batches)
                            rows.append(row)
                        tensors.update(values)
                        if reference is not None:
                            for rid, value in values.items():
                                key = rid.split(':', 1)[1] if meta['stage'] == 'smoke' else rid
                                errors[rid] = error_metrics(value, reference[key][:len(value)], config['numerics'])
                    print(label, suite['name'], case['case_id'], 'completed', flush=True)
            torch.save(tensors, path)
        finally:
            if scheduler is not None:
                scheduler.shutdown()
    all_errors = [e for values in errors.values() for e in values]
    passed = not all_errors or all(e['passed'] for e in all_errors)
    result = dict(kind=kind, mode=mode, variant=variant, status='completed', passed=passed,
        reference_scope='external_HF_FP32' if mode == 'mixed' else 'within_native_BF16_strategy_invariance',
        rows=rows, errors=errors, observed=observed, vectors=sum(len(t) for t in tensors.values()),
        tensor=dict(path=str(path.relative_to(ROOT)), sha256=digest(path)), performance_allowed=False,
        precision=dict(tf32=False, bf16_reduced_reduction=False, fp16_reduced_reduction=False),
        packages={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'flashinfer-python', 'sgl-kernel')},
        gpu_end=telemetry())
    write(directory/f'{label}.json', result)
    print(label, 'passed', passed, 'max_relative_rmse', max((e['relative_rmse'] for e in all_errors), default=0.), flush=True)
    return 0 if meta['stage'] != 'smoke' or passed else 1


def service_worker(directory, mode, variant):
    meta = read(directory/'provenance.json')
    check_sources(meta)
    os.environ['ROUTESCALE_I2_PRECISION_MODE'] = mode
    from scripts.run_i2_service_gate import worker
    config = dict(meta['config'], results_dir=str(directory.relative_to(ROOT)),
        runtime_dir=str(Path(meta['config']['runtime_dir'])/directory.name),
        service_launcher='scripts/launch_i2_precision_service.py')
    return worker(config, mode, variant)


def benchmark_worker(directory, mode, variant, run):
    sys.path.insert(0, str(ROOT/'runs/vendor/mini-sglang-i2/python'))
    import torch
    import serving.i2_runtime as runtime
    from serving.i2_precision_runtime import PrecisionScheduler
    from serving.i2_precision_policy import POLICIES
    from serving.i2_worker import replay, telemetry
    meta = read(directory/'provenance.json')
    check_sources(meta)
    precision_setup()
    config = meta['config']
    workload = read(directory/'workload.json')
    label = f'benchmark_{mode}_{variant}_run{run}'
    output = directory/f'{label}.json'
    if output.exists():
        raise ValueError('Preserve timing result')
    raw = ROOT/config['runtime_dir']/directory.name/label
    raw.mkdir(parents=True, exist_ok=False)
    validation = ROOT/meta['validation_dir']
    validation.mkdir(parents=True, exist_ok=True)
    observed = []
    def factory(*args, **kwargs):
        scheduler = PrecisionScheduler(*args, **kwargs, mode=mode)
        observed.append(scheduler.engine._i2_precision_observed)
        return scheduler
    record = dict(status='fail', mode=mode, variant=variant, run_id=run,
        config_sha256=meta['config_sha256'], workload_sha256=meta['workload_sha256'], patched_tree=meta['series']['tree'],
        runtime_dir=str(raw.relative_to(ROOT)), policy=POLICIES[mode],
        precision=dict(tf32=False, bf16_reduced_reduction=False, fp16_reduced_reduction=False), gpu_before=telemetry())
    try:
        with patch.object(runtime, 'ReplayScheduler', factory), torch.inference_mode():
            record['replay'] = replay(config, workload, variant, 'benchmark', validation, raw, run)
        record.update(status=record['replay']['status'], observed=observed[0], gpu_end=telemetry(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        record['error'] = traceback.format_exc()
        print(record['error'], flush=True)
    write(output, record)
    print(label, record['status'], flush=True)
    return 0 if record['status'] == 'pass' else 1


def freeze(directory, config, stage, workload):
    from scripts.prepare_i2_minisglang import validate
    directory.mkdir(parents=True, exist_ok=False)
    names = ['config/serving/i2_precision.json', config['base_config'], 'docs/i2_precision_protocol.md',
        'scripts/run_i2_precision.py', 'serving/i2_precision_policy.py', 'serving/i2_precision_runtime.py',
        'serving/i2_runtime.py', 'serving/i2_worker.py', 'serving/i2_workloads.py',
        'serving/i2_numerical_policy.py', 'serving/i2_linear_tile_policy.py',
        'scripts/run_i2_single_gpu.py', 'scripts/prepare_i2_minisglang.py', 'tests/test_i2_precision_policy.py']
    if stage in {'http', 'performance'}:
        names += ['scripts/launch_i2_precision_service.py', 'scripts/launch_i2_service.py',
            'scripts/run_i2_service_gate.py', 'serving/minisgl_observer.py', 'serving/online_client.py',
            'scripts/run_i1_lifecycle.py', 'scripts/run_i1_repaired.py']
    if stage == 'performance':
        names += ['scripts/audit_i2_precision.py']
    validation = ROOT/config['validation_dir']/directory.name
    validation.mkdir(parents=True, exist_ok=False)
    write(directory/'workload.json', workload)
    write(directory/'config.json', config)
    meta = dict(stage=stage, config=config, config_sha256=token_hash(config),
        source_sha256={n: digest(ROOT/n) for n in names}, series=validate(ROOT/config['checkout']),
        model_sha256={p.name: digest(p) for p in (ROOT/config['model_path']).iterdir() if p.is_file()},
        validation_dir=str(validation.relative_to(ROOT)), workload_sha256=digest(directory/'workload.json'),
        paid_resources_used=False, performance_allowed=False)
    if stage == 'smoke':
        ref = ROOT/'runs/validation/i2_single_gpu_v1/attempt001/reference.pt'
        meta['original_reference'] = dict(path=str(ref.relative_to(ROOT)), sha256=digest(ref))
    for name in names:
        dest = directory/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    write(directory/'provenance.json', meta)
    return meta


def prerequisites(config):
    q = ROOT/config['qualification_directory']
    l = ROOT/config['lifecycle_directory']
    qa, qm = read(q/'audit.json'), read(q/'provenance.json')
    la, lm = read(l/'audit.json'), read(l/'provenance.json')
    if not qa['qualification_passed'] or not la['HTTP_lifecycle_passed']:
        raise ValueError('Independent numerical and HTTP gates required')
    if lm['qualification_audit_sha256'] != digest(q/'audit.json'):
        raise ValueError('HTTP does not bind qualified numerics')
    for meta in (qm, lm):
        if meta['config'] != config:
            raise ValueError('Qualified config changed')
        check_sources(meta)
    if qa['auditor_sha256'] != digest(ROOT/'scripts/audit_i2_precision.py') or la['auditor_sha256'] != digest(ROOT/'scripts/audit_i2_precision.py'):
        raise ValueError('Gate auditor changed')
    return dict(performance_allowed=True, qualification_audit_sha256=digest(q/'audit.json'),
        lifecycle_audit_sha256=digest(l/'audit.json'), modes=config['modes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke', 'qualify', 'http', 'performance'])
    parser.add_argument('--attempt')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--kind', choices=['hf_reference', 'native_reference', 'numeric', 'service', 'benchmark'])
    parser.add_argument('--mode', default='mixed', choices=['mixed', 'bf16_control'])
    parser.add_argument('--variant', default='naive_full')
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        if args.kind == 'service':
            return service_worker(args.worker, args.mode, args.variant)
        if args.kind == 'benchmark':
            return benchmark_worker(args.worker, args.mode, args.variant, args.run)
        return qualification_worker(args.worker, args.mode, args.variant, args.kind)
    if not args.stage or not args.attempt or not args.attempt.replace('_', '').isalnum():
        raise ValueError('Explicit stage and fresh simple attempt required')
    lock = (ROOT/'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load_config()
    suites = read(ROOT/config['suite_source'])
    workload = suites if args.stage in {'smoke', 'qualify', 'http'} else suites['suites'][0]['workload']
    if args.stage == 'smoke':
        workload = dict(suites=suites['suites'][:1])
    if args.stage == 'http':
        qa = read(ROOT/config['qualification_directory']/'audit.json')
        if not qa['qualification_passed']:
            raise ValueError('Numerical qualification first')
    gate = prerequisites(config) if args.stage == 'performance' else None
    directory = ROOT/config['results_dir']/args.attempt
    meta = freeze(directory, config, args.stage, workload)
    if args.stage == 'http':
        meta['qualification_audit_sha256'] = digest(ROOT/config['qualification_directory']/'audit.json')
        write(directory/'provenance.json', meta)
    if gate:
        write(directory/'gate.json', gate)
    from scripts.run_i2_single_gpu import launch
    journal = []
    def call(kind, mode='mixed', variant='naive_full', run=0):
        label = kind if kind in {'hf_reference', 'native_reference'} else f'{kind}_{mode}_{variant}' + (f'_run{run}' if run else '')
        launch([sys.executable, __file__, '--worker', str(directory), '--kind', kind, '--mode', mode,
            '--variant', variant, '--run', str(run)], label, config, args.attempt, directory, journal)
    if args.stage == 'smoke':
        for variant in ('radix_graph124', 'radix_overlap'):
            call('numeric', variant=variant)
    elif args.stage == 'qualify':
        call('hf_reference')
        call('native_reference', mode='bf16_control')
        for mode in config['modes']:
            for variant in config['variants']:
                call('numeric', mode, variant)
    elif args.stage == 'http':
        for mode in config['modes']:
            for variant in config['variants']:
                call('service', mode, variant)
    else:
        names = list(config['variants'])
        for run in range(1, 4):
            order = names if run == 1 else list(reversed(names)) if run == 2 else names[3:] + names[:3]
            for index, variant in enumerate(order):
                modes = config['modes'] if run == 1 or (run == 3 and index % 2 == 0) else list(reversed(config['modes']))
                for mode in modes:
                    call('benchmark', mode, variant, run)


if __name__ == '__main__':
    raise SystemExit(main())
