"""Serial external-reference alignment; immutable diagnostics, never timing admission."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ('hf_fp32', 'hf_bf16', 'native_bf16', 'residual_fp32', 'native_fp16', 'fp32_math')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def worker(directory, mode):
    sys.path.insert(0, str(ROOT / 'runs/vendor/mini-sglang-i2/python'))
    import torch
    from serving.i2_worker import error_metrics, telemetry
    from serving.i2_workloads import digest
    from serving.i2_external_alignment import HFTrace, install_precision_intervention
    meta = json.loads((directory / 'provenance.json').read_text())
    config = meta['config']
    for name, sha in meta['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Frozen source changed: ' + name)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    workload = json.loads((directory / 'workload.json').read_text())
    frozen = torch.load(ROOT / meta['reference']['path'], mmap=True, weights_only=True)
    validation = ROOT / meta['validation_dir']
    scheduler = observer = restore = None
    rows, all_values, trace, native_waves = [], {}, {}, []
    traced_rid = None
    observed = {}
    with torch.inference_mode():
        try:
            if mode.startswith('hf_'):
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(ROOT / config['model_path'], local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation='eager')
                if mode == 'hf_fp32':
                    model = model.float()
                model = model.cuda().eval()
                observed = dict(weight_dtype=str(model.model.embed_tokens.weight.dtype),
                    activation_dtype=str(model.model.embed_tokens.weight.dtype), attention_execution='HF eager')
            else:
                from serving import i2_runtime as runtime
                from serving.i2_layer_trace import LayerTrace
                native_config = runtime.engine_config
                def selected_config(cfg, variant):
                    result = native_config(cfg, variant)
                    if mode == 'native_fp16':
                        result = replace(result, dtype=torch.float16)
                    return result
                # Explicit construction-only override; existing runtime source is unchanged.
                with patch.object(runtime, 'engine_config', selected_config):
                    scheduler = runtime.ReplayScheduler(config, meta['variant'], capture_logits=True)
                if mode in {'residual_fp32', 'fp32_math'}:
                    restore = install_precision_intervention(scheduler.engine, mode)
                observed = dict(weight_dtype=str(scheduler.engine.model.model.embed_tokens.weight.dtype),
                    kv_dtype=str(scheduler.engine.kv_cache.dtype), engine_declared_dtype=str(scheduler.engine.dtype),
                    attention_execution='PyTorch FP32 over actual pages' if mode == 'fp32_math' else 'FlashInfer fa2',
                    graph_buckets=scheduler.engine.graph_runner.graph_bs_list,
                    residual_dtype='torch.float32' if mode in {'residual_fp32', 'fp32_math'} else str(scheduler.engine.dtype))
            for case in workload['cases']:
                if scheduler is not None:
                    runtime.flush_cache(scheduler, poison=True)
                for phase in (('cold',) if meta['scope'] == 'pilot' else ('cold', 'warm')):
                    requests = case['waves'][phase]
                    if meta['scope'] == 'pilot':
                        requests = [dict(r, continuation_ids=r['continuation_ids'][:4]) for r in requests]
                    if scheduler is None:
                        tensors = {}
                        for request in requests:
                            rid = request['request_id']
                            if traced_rid is None:
                                observer = HFTrace(model)
                                traced_rid = rid
                            inputs = torch.tensor([request['input_ids']], device='cuda')
                            past, values = None, []
                            for token in request['continuation_ids']:
                                out = model(input_ids=inputs, past_key_values=past, use_cache=True, logits_to_keep=1)
                                values.append(out.logits[0, -1].detach().clone())
                                past = out.past_key_values
                                if observer is not None:
                                    observer.enabled = False
                                inputs = torch.tensor([[token]], device='cuda')
                            tensors[rid] = torch.stack(values).float().cpu().contiguous()
                            if observer is not None:
                                trace = observer.cpu()
                                observer.close()
                                observer = None
                            del out, past, values
                        released = None
                    else:
                        if traced_rid is None and meta['variant'] in {'naive_full', 'radix_full'}:
                            traced_rid = requests[0]['request_id']
                            observer = LayerTrace(scheduler, prefix=0, suffix=128)
                            observer.phase = 'warm'  # Observe this single cold prefill, no cache hit.
                        row, tensors, batches = scheduler.wave(requests)
                        released = row['released']
                        settings = config['variants'][meta['variant']]
                        expected_hit = config['shared_prefix_tokens'] if phase == 'warm' and case['prefix_mode'] == 'shared' and settings['cache'] == 'radix' else 0
                        if phase == 'warm' and [r['cached_tokens'] for r in row['requests']] != [expected_hit] * len(requests):
                            raise ValueError('Actual warm prefix hit contract changed')
                        native_waves.append(dict(case_id=case['case_id'], phase=phase, requests=row['requests'], batches=batches))
                        if observer is not None:
                            trace = observer.cpu()['ops']
                            observer.close()
                            observer = None
                    for rid, actual in tensors.items():
                        metrics = error_metrics(actual, frozen[rid][:len(actual)], config['numerics'])
                        rows.append(dict(request_id=rid, case_id=case['case_id'], phase=phase,
                            errors=metrics, released=released))
                    all_values.update(tensors)
                print(mode, case['case_id'], 'completed', flush=True)
            torch.save(all_values, validation / f'{mode}.pt')
            torch.save(trace, validation / f'{mode}_trace.pt')
        finally:
            if observer is not None:
                observer.close()
            if restore is not None:
                restore()
            if scheduler is not None:
                scheduler.shutdown()
    errors = [e for r in rows for e in r['errors']]
    artifacts = {name: dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for name, path in
                 [('logits', validation / f'{mode}.pt'), ('trace', validation / f'{mode}_trace.pt')]}
    record = dict(mode=mode, variant=meta['variant'], status='completed', performance_allowed=False,
        observed=observed,
        native_waves=native_waves,
        evidence_class='external_reference_precision_diagnostic', rows=rows, tensors=artifacts,
        traced_request=traced_rid, trace_rows=128, trace_scope='first_cold_prefill',
        vectors=len(errors), passed_vectors=sum(e['passed'] for e in errors),
        max_relative_rmse=max(e['relative_rmse'] for e in errors), max_abs=max(e['max_abs'] for e in errors),
        numerical_passed=all(e['passed'] for e in errors), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        packages={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'flashinfer-python', 'sgl-kernel')},
        precision=dict(tf32=False, bf16_reduced_reduction=False, fp16_reduced_reduction=False), gpu_end=telemetry())
    write(directory / f'{mode}.json', record)
    print(mode, record['passed_vectors'], '/', len(errors), record['max_relative_rmse'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt')
    parser.add_argument('--scope', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--variant', choices=['naive_full', 'radix_full', 'radix_chunk256', 'radix_chunk128'], default='naive_full')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.mode)
    if not args.attempt or not args.attempt.replace('_', '').isalnum():
        raise ValueError('Fresh simple attempt name required')
    lock = (ROOT / 'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from serving.i2_workloads import CONFIG, digest
    from scripts.run_i2_single_gpu import launch
    from scripts.prepare_i2_minisglang import validate
    config = json.loads(CONFIG.read_text())
    directory = ROOT / 'results/i_minisglang/i2_external_alignment_v1' / args.attempt
    validation = ROOT / 'runs/validation/i2_external_alignment_v1' / args.attempt
    directory.mkdir(parents=True, exist_ok=False)
    validation.mkdir(parents=True, exist_ok=False)
    names = ['scripts/diagnose_i2_external_alignment.py', 'serving/i2_external_alignment.py',
             'serving/i2_runtime.py', 'serving/i2_layer_trace.py', 'serving/i2_worker.py', 'serving/i2_workloads.py',
             'scripts/run_i2_single_gpu.py', 'scripts/audit_i2_external_alignment.py',
             'tests/test_i2_external_alignment.py', 'docs/i2_external_alignment_protocol.md', str(CONFIG.relative_to(ROOT))]
    hf_source = ROOT / 'runs/envs/mini-sglang/lib/python3.11/site-packages/transformers/models/qwen3/modeling_qwen3.py'
    reference = ROOT / config['validation_dir'] / 'attempt001/reference.pt'
    from safetensors import safe_open
    checkpoint_dtypes = set()
    for file in (ROOT / config['model_path']).glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as handle:
            checkpoint_dtypes.update(handle.get_slice(n).get_dtype() for n in handle.keys())
    if checkpoint_dtypes != {'BF16'}:
        raise ValueError('FP16 intervention requires exactly BF16-rounded checkpoint values')
    meta = dict(stage='I2', scope=args.scope, variant=args.variant, modes=args.modes, config=config,
        source_sha256={n: digest(ROOT / n) for n in names}, series=validate(ROOT / config['checkout']),
        checkpoint_dtypes=sorted(checkpoint_dtypes), hf_source_sha256=digest(hf_source),
        model_sha256={p.name: digest(p) for p in (ROOT / config['model_path']).iterdir() if p.is_file()},
        reference=dict(path=str(reference.relative_to(ROOT)), sha256=digest(reference)),
        workload_sha256=digest(ROOT / config['workload_path']), validation_dir=str(validation.relative_to(ROOT)),
        performance_allowed=False, paid_resources_used=False)
    for name in names:
        dest = directory / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(hf_source, directory / 'source_snapshot/modeling_qwen3.py')
    shutil.copy2(ROOT / config['workload_path'], directory / 'workload.json')
    write(directory / 'provenance.json', meta)
    journal = []
    runtime = dict(config, runtime_dir='runs/logs/i_minisglang/i2_external_alignment_v1', subprocess_timeout_s=900)
    for mode in args.modes:
        launch([sys.executable, __file__, '--worker', str(directory), '--mode', mode], mode,
               runtime, args.attempt, directory, journal)


if __name__ == '__main__':
    main()
