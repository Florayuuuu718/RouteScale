"""Fresh qualification and serial, balanced H3 v5 timing replication."""
from __future__ import annotations

import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_performance import SOURCES, VARIANTS, now, value_hash
from serving.h3_stability import CONFIG, EXTRA_SOURCES, admission, check_sources, config, gpu_idle, plan, read, require_ac, sensors


def freeze(c, directory):
    if directory.exists():
        raise ValueError('Fresh result directory required; no overwriting')
    old = read(ROOT / c['base_config']); old_dir = ROOT / old['results_dir']
    gate = read(old_dir / 'gate.json')
    if gate['status'] != 'pass' or not gate['performance_execution_allowed']:
        raise ValueError('Original v5 qualification is not accepted')
    names = list(dict.fromkeys((*SOURCES, *EXTRA_SOURCES)))
    source_sha = {}
    for name in names:
        p = directory / 'source_snapshot' / name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, p); source_sha[name] = digest(p)
    frozen_old = read(old_dir / 'source_snapshot/manifest.json')['source_sha256']
    if any(digest(ROOT / name) != sha for name, sha in frozen_old.items()):
        raise ValueError('Original v5 source changed')
    evidence = [ROOT / c['base_config'], ROOT / old['workload_path'], ROOT / old['artifact_manifest'],
        ROOT / old['data_path'], ROOT / old['tokenizer_path'], old_dir / 'gate.json',
        old_dir / 'audit.json', old_dir / 'preparation.json', old_dir / 'reference.json']
    manifest = read(ROOT / old['artifact_manifest']); evidence.append(ROOT / manifest['artifact_path'])
    cases = [r for r in read(ROOT / old['workload_path'])['cases'] if r['case_id'] in c['case_ids']]
    meta = dict(config=c, source_sha256=source_sha, evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},
        base_provenance=gate['provenance'], base_gate_sha256=digest(old_dir / 'gate.json'), cases=cases,
        protocol_date='2026-11-02', AC=require_ac(c))
    write_result(meta, directory / 'provenance.json')
    historical = {str(p.relative_to(ROOT)): digest(p) for base in ('g_serving', 'h_nanovllm', 'i_minisglang')
        for p in (ROOT / 'results' / base).rglob('*.json') if not p.is_relative_to(directory)}
    write_result(historical, directory / 'historical.json')
    write_result(plan(c), directory / 'plan.json')
    cpu = read(ROOT / 'runs/tmp/h3_stability_cpu_tests.json')
    if (not cpu['passed'] or cpu['tests'] != 23 or digest(ROOT / cpu['log']) != cpu['log_sha256']
            or any(digest(ROOT / n) != sha for n,sha in cpu['source_sha256'].items())):
        raise ValueError('Current CPU regression evidence required before qualification')
    shutil.copy2(ROOT / cpu['log'], directory / 'cpu_tests.txt')
    cpu['log'] = str((directory / 'cpu_tests.txt').relative_to(ROOT))
    write_result(cpu, directory / 'cpu_tests.json')


def worker(c, directory, mode, label):
    import torch
    from serving.h3_performance import create_path, load_inputs, run_wave, tensor_metadata
    meta = check_sources(directory)
    old, artifact, cases, provenance = load_inputs(ROOT / c['base_config'])
    if provenance != meta['base_provenance']:
        raise ValueError('Original v5 GPU/environment provenance differs')
    require_ac(c)
    if mode == 'qualification':
        output = directory / 'qualification' / f'{label}.json'
        if output.exists(): raise ValueError('Preserve qualification')
        report = dict(status='running', variant=label, base_provenance=provenance,
            replication_source_sha256=meta['source_sha256'], cases={})
        try:
            for case in cases:
                if case['case_id'] not in c['case_ids']: continue
                session = create_path(artifact, case, old, label)
                try:
                    trace, tensor, usage = run_wave(session, case, 'model_execution', collect_logits=True, poison=True)
                    target = ROOT / c['validation_dir'] / f'{label}_{case["case_id"]}.pt'
                    if target.exists(): raise ValueError('Preserve qualification tensor')
                    q = tensor_metadata(tensor, target)
                    report['cases'][case['case_id']] = dict(**q, trace=trace, usage=usage,
                        weight_sha256=session.weight_sha256, kv_bytes=session.kv_bytes)
                    del tensor
                finally: session.close()
                gc.collect(); torch.cuda.empty_cache()
            report['status'] = 'pass'
        except BaseException:
            report.update(status='fail', traceback=traceback.format_exc()); raise
        finally: write_result(report, output)
        print('H3 fresh qualification:', label, 'pass', flush=True)
        return
    gate = read(directory / 'qualification_gate.json')
    if not gate['passed']: raise ValueError('New qualification required')
    job = next(r for r in plan(c) if r['label'] == label)
    case = next(r for r in cases if r['case_id'] == job['case_id'])
    output = directory / 'raw' / f'{label}.json'
    if output.exists(): raise ValueError('Preserve formal process')
    report = dict(stage='H3', status='running', job=job, base_provenance=provenance,
        replication_source_sha256=meta['source_sha256'], gate_sha256=digest(directory / 'qualification_gate.json'),
        workload_sha256=case['workload_sha256'], dtype='bfloat16', prefix_cache=False, token_policy=old['token_policy'],
        warmup_repeats=c['warmup_repeats'], measure_repeats=c['measure_repeats'],
        measurements={}, sensors=[], started_at_utc=now(), pid=os.getpid())
    session = None
    try:
        session = create_path(artifact, case, old, job['variant'])
        report.update(weight_sha256=session.weight_sha256, kv_bytes=session.kv_bytes, initialization=session.initialization,
            actual_reduction_permission=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            graph_keys=sorted(session.engine.model_runner.graphs) if job['variant'] == 'paged_graph' else [])
        for scope in job['scopes']:
            require_ac(c); report['sensors'].append(sensors(scope + '/before_warmup'))
            warm = [run_wave(session, case, scope)[0] for _ in range(c['warmup_repeats'])]
            torch.cuda.synchronize(); report['sensors'].append(sensors(scope + '/after_warmup'))
            torch.cuda.reset_peak_memory_stats()
            traces = []
            for index in range(c['measure_repeats']):
                traces.append(run_wave(session, case, scope)[0])
                if (index + 1) % c['sensor_every_waves'] == 0:
                    report['sensors'].append(sensors(f'{scope}/after_measure_{index + 1}'))
            report['measurements'][scope] = dict(traces=traces, warmup_traces=warm,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['free_bytes_after_run'] = torch.cuda.mem_get_info()[0]
        report['status'] = 'pass'
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc()); raise
    finally:
        if session is not None: session.close()
        report['finished_at_utc'] = now(); write_result(report, output)
    print('H3 replication formal:', label, 'pass', flush=True)


def child(command, label, directory, c, rows, env):
    require_ac(c); before = gpu_idle()
    log = ROOT / 'runs/logs/h_nanovllm/h3_timing_replication_v1' / f'{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    row = dict(label=label, command=command, status='running', gpu_before=before, log=str(log.relative_to(ROOT)), started_at_utc=now())
    rows.append(row)
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        row['pid'] = process.pid; write_result(rows, directory / 'processes.json')
        started = time.monotonic(); escalations = []
        try:
            rc = process.wait(timeout=c['subprocess_timeout_seconds'])
        except BaseException:
            for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                if process.poll() is not None: break
                escalations.append(sig.name)
                try: os.killpg(process.pid, sig)
                except ProcessLookupError: break
                try: process.wait(timeout=timeout)
                except subprocess.TimeoutExpired: pass
            row.update(returncode=process.poll(), status='fail', escalations=escalations)
            write_result(rows, directory / 'processes.json'); raise
    row.update(returncode=rc, duration_s=time.monotonic() - started, log_sha256=digest(log),
        finished_at_utc=now(), escalations=escalations)
    try:
        row['gpu_after'] = gpu_idle()
        if rc != 0: raise ValueError(f'Child failed: {label}, rc={rc}; see {log}')
        row['status'] = 'pass'
    finally: write_result(rows, directory / 'processes.json')
    print('completed:', label, 'seconds:', round(row['duration_s'], 2), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['qualification', 'measurement'], required=True)
    p.add_argument('--worker', action='store_true'); p.add_argument('--label')
    args = p.parse_args(); c = config(); directory = ROOT / c['results_dir']
    if args.worker:
        worker(c, directory, args.stage, args.label); return
    lock_path = ROOT / 'runs/tmp/h2_runner.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); gpu_idle(); require_ac(c)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
            TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'runs/cache/h3_timing_replication_v1/inductor'),
            TRITON_CACHE_DIR=str(ROOT / 'runs/cache/h3_timing_replication_v1/triton'))
        python = str(ROOT / 'runs/envs/nano-vllm/bin/python')
        if args.stage == 'qualification':
            freeze(c, directory); rows = []
            old_check = child([python, 'scripts/audit_h3_performance.py', '--config', c['base_config'], '--check-only'],
                'original_v5_readonly_audit', directory, c, rows, env)
            write_result(old_check, directory / 'baseline_check.json')
            for variant in VARIANTS:
                child([python, str(Path(__file__).resolve()), '--stage', 'qualification', '--worker', '--label', variant],
                    'qualification_' + variant, directory, c, rows, env)
        else:
            admission(directory)
            rows = read(directory / 'processes.json')
            if any(r['label'].startswith('formal_') for r in rows): raise ValueError('Preserve measurement attempt')
            for index, job in enumerate(plan(c)):
                check_sources(directory)
                print(f'H3 formal {index + 1}/36: {job["label"]}', flush=True)
                child([python, str(Path(__file__).resolve()), '--stage', 'measurement', '--worker', '--label', job['label']],
                    'formal_' + job['label'], directory, c, rows, env)
        check_sources(directory); print('H3 stage complete:', args.stage, flush=True)


if __name__ == '__main__': main()
