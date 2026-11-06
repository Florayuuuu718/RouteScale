"""Sequential host-bound F/H2/I2 execution, with a wall-time/cost ceiling."""
from __future__ import annotations

import fcntl
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from rental.common import ROOT, output, positive, read, sha, source_hashes, validate_hashes, write
from rental.deployment import environment_report, verify_assets


def plan(directory, root=ROOT):
    directory = Path(directory).absolute()
    mini = str(root / 'runs/envs/mini-sglang/bin/python')
    training = str(root / '.venv/bin/python')
    python = sys.executable
    tp = str(directory / 'tp')
    f = str(directory / 'f')
    steps = []

    def add(name, command, timeout, phase='qualification'):
        steps.append(dict(name=name, command=list(map(str, command)), timeout_s=timeout, phase=phase))

    for world in (1, 2, 4):
        add(f'nccl{world}', [python, 'scripts/prepare_serving_tp.py', '--stage', 'collective', '--world', world,
                            '--directory', directory / f'nccl{world}'], 960)
    add('tp_qualification', [python, 'scripts/prepare_serving_tp.py', '--stage', 'qualify', '--directory', tp], 32 * 920)
    add('tp_audit', [mini, 'scripts/audit_serving_tp.py', '--directory', tp], 1800)
    add('f_module', [python, 'scripts/prepare_f_ep.py', '--host-qualification', '--execute', '--python', training,
                     '--output', directory / 'f/module'], 8 * 1240)
    add('f_training', [python, 'scripts/prepare_f_training.py', '--host-qualification', '--execute', '--python', training,
                       '--output', directory / 'f/training'], 5 * 1240)
    add('tp_benchmark', [python, 'scripts/prepare_serving_tp.py', '--stage', 'benchmark', '--directory', tp], 90 * 920, 'performance')
    add('f_benchmark', [python, 'scripts/prepare_f_training.py', '--benchmark', '--host-root', f, '--execute',
                        '--python', training, '--output', directory / 'f/performance'], 9 * 920, 'performance')
    add('tp_trace', [python, 'scripts/prepare_serving_tp.py', '--stage', 'trace', '--directory', tp], 30 * 920, 'trace')
    add('f_trace', [python, 'scripts/collect_f_ep_trace.py', '--host-root', f, '--python', training,
                    '--output', directory / 'f/trace'], 1240, 'trace')
    return dict(schema_version=1, steps=steps, gpu_count=4,
                counts=dict(tp_qualification=30, tp_performance=90, tp_trace=30,
                            f_module_qualification=8, f_training_qualification=5, f_performance=9, f_trace=1),
                worst_case_timeout_s=sum(s['timeout_s'] for s in steps),
                warning='Timeout sums are a failure bound, not a rental duration or cost estimate.',
                scopes={'H2': 'nano-vLLM TP1/2/4 Qwen3-0.6B', 'I2': 'Mini-SGLang TP1/2/4 Qwen3-0.6B',
                        'F': 'EP2 complete MoE training; manual replicated control; module communication trace'},
                unplanned=['C1-C7 already accepted', 'larger serving model', 'multigpu HTTP lifecycle/I3',
                           'native PyTorch DDP reducer MoE timing', 'full GPT EP communication attribution'])


def host_budget(config, hostname, uuids):
    if config.get('existing_host_confirmed') is not True or config.get('hostname') != hostname:
        raise ValueError('Existing host and execution budget must be explicitly filled in')
    if len(uuids) != 4 or len(set(uuids)) != 4 or config.get('gpu_uuids') != uuids:
        raise ValueError('Exactly four recorded GPU UUIDs, in physical index order, are required')
    runtime = positive(config.get('max_runtime_s'))
    rate = positive(config.get('hourly_rate_all_gpus'))
    cost = positive(config.get('max_cost'))
    if runtime < 60 or not config.get('currency') or rate * runtime / 3600 > cost:
        raise ValueError('Runtime must fit the declared all-GPU cost ceiling')
    return runtime


def gpu_rows():
    text = output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    rows = [line.split(', ') for line in text.splitlines()]
    if any(int(row[0]) != i for i, row in enumerate(rows)):
        raise ValueError('Physical GPU index order changed')
    if any(float(r[4]) != 0 or float(r[5]) != 0 for r in rows):
        raise ValueError('All selected GPUs must be idle')
    if len({(r[2], r[3]) for r in rows}) != 1:
        raise ValueError('Homogeneous GPU model/memory required')
    return rows


def process_table():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(pid, known):
    table = process_table()
    parents = {pid} | {p for p, birth in known.items() if p in table and table[p][1] == birth}
    while True:
        new = {p for p, (parent, _, _) in table.items() if parent in parents}
        if new <= parents:
            break
        parents |= new
    known.update({p: table[p][1] for p in parents if p in table})
    return {p for p, birth in known.items() if p in table and table[p][1] == birth and table[p][2] != 'Z'}


def run_command(command, log, timeout, root=ROOT):
    """Track descendants even when legacy launchers start separate sessions."""
    start = time.monotonic()
    known = {}
    process = None
    failure = None
    cleanup = False
    # Adopt orphaned grandchildren from legacy start_new_session launchers, so
    # a controller that exits before the next poll cannot hide a live worker.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError('Linux child-subreaper support required')
    initial_children = {pid for pid, (parent, _, _) in process_table().items() if parent == os.getpid()}

    def track():
        table = process_table()
        known.update({pid: birth for pid, (parent, birth, _) in table.items()
                      if parent == os.getpid() and pid not in initial_children})
        return descendants(process.pid, known)
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with Path(log).open('x') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while True:
                    track()
                    rc = process.poll()
                    if rc is not None:
                        if rc:
                            failure = f'exit {rc}'
                        break
                    if time.monotonic() - start >= timeout:
                        failure = 'timeout'
                        break
                    time.sleep(0.05)
            except BaseException as exc:
                failure = type(exc).__name__
            finally:
                # Give normal resource trackers a brief chance to exit before
                # escalating, then stop only processes whose start IDs match.
                grace = time.monotonic() + (0 if failure else 1)
                while track() and time.monotonic() < grace:
                    time.sleep(.05)
                alive = track()
                if alive:
                    cleanup = True
                    failure = failure or 'descendants remained after exit'
                    for sig, wait_s in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                        for pid in track():
                            try:
                                os.kill(pid, sig)
                            except ProcessLookupError:
                                pass
                        until = time.monotonic() + wait_s
                        while track() and time.monotonic() < until:
                            time.sleep(.05)
                process.wait(timeout=2)
                for pid in known:
                    if pid != process.pid:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return dict(passed=failure is None, failure=failure, returncode=process.returncode,
                elapsed_s=time.monotonic() - start, cleanup_required=cleanup,
                descendants_remaining=sorted(descendants(process.pid, known)), log_sha256=sha(log))


def preflight(config, directory):
    rows = gpu_rows()
    runtime = host_budget(config, socket.gethostname(), [r[1] for r in rows])
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0,1,2,3'):
        raise ValueError('Unset CUDA_VISIBLE_DEVICES; launchers use physical indices 0..3')
    if output(['git', 'status', '--porcelain']):
        raise ValueError('Clean source checkout required')
    verify_assets(ROOT)
    deployment = read(ROOT / 'runs/rental/deployment.json')
    validate_hashes(ROOT, deployment['assets_sha256'])
    environments = environment_report(ROOT)
    from serving.delivery import doctor, load
    doctor(load())
    toolkit_version = output([ROOT / 'runs/toolchains/cuda-13.0-i1/bin/nvcc', '--version'])
    if 'release 13.0' not in toolkit_version:
        raise ValueError('CUDA 13.0 compiler required')
    for name, path in (('training', '.venv'), ('mini', 'runs/envs/mini-sglang'), ('nano', 'runs/envs/nano-vllm')):
        result = run_command([str(ROOT / path / 'bin/python'), 'scripts/check_rental_cuda.py',
                              '--output', str(directory / f'{name}_cuda.json')], directory / f'{name}_cuda.log', 120)
        if not result['passed']:
            raise ValueError(f'{name} CUDA software preflight failed')
    from serving.tp_preparation import inventory
    inventory(directory / 'hardware')
    report = dict(passed=True, hostname=socket.gethostname(), gpu_rows=rows, environments=environments,
                  nvcc=toolkit_version,
                  deployment_sha256=sha(ROOT / 'runs/rental/deployment.json'),
                  max_runtime_s=runtime, source_sha256=source_hashes(), commit=output(['git', 'rev-parse', 'HEAD']))
    write(directory / 'preflight.json', report)
    return report


def resume_state(directory, config):
    journal = read(directory / 'session.json')
    if not journal['passed'] or journal['status'] != 'completed':
        raise ValueError('Failed/partial stages are preserved; automatic retries are disabled')
    if read(directory / 'host_contract.json') != config:
        raise ValueError('A resumed session must retain its host and budget')
    for file, key in (('plan.json', 'plan_sha256'), ('preflight.json', 'preflight_sha256'),
                      ('host_contract.json', 'host_contract_sha256')):
        if sha(directory / file) != journal[key]:
            raise ValueError('Changed session manifest')
    planned = [s['name'] for s in read(directory / 'plan.json')['steps']]
    if [s['name'] for s in journal['steps']] != planned[:len(journal['steps'])]:
        raise ValueError('Completed steps must form a contiguous plan prefix')
    for step in journal['steps']:
        if not step['passed'] or sha(directory / f'{step["name"]}.log') != step['log_sha256']:
            raise ValueError('Changed or failed completed stage')
    validate_hashes(ROOT, read(directory / 'preflight.json')['source_sha256'])
    return journal


def execute(directory, config, through='trace', resume=False):
    directory = Path(directory).absolute()
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    lockpath = ROOT / 'runs/tmp/rental.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Validate cost/host before any CUDA initialization or expensive preflight.
        runtime = host_budget(config, socket.gethostname(), [r[1] for r in gpu_rows()])
        if resume:
            journal = resume_state(directory, config)
            pipeline = read(directory / 'plan.json')
            phases = ['qualification', 'performance', 'trace']
            if phases.index(through) < phases.index(journal['completed_through']):
                raise ValueError('Resume cannot downgrade the completed phase')
            if time.time() >= journal['started_unix'] + runtime - 10:
                raise TimeoutError('Original session deadline expired; resume cannot reset the budget')
        else:
            write(directory / 'host_contract.json', config)
            pipeline = plan(directory)
            write(directory / 'plan.json', pipeline)
            journal = dict(passed=False, status='preflight', steps=[], completed_through=None, started_unix=time.time(),
                           host_contract_sha256=sha(directory / 'host_contract.json'), plan_sha256=sha(directory / 'plan.json'))
            write(directory / 'session.json', journal)
        try:
            if resume:
                if environment_report(ROOT) != read(directory / 'preflight.json')['environments']:
                    raise ValueError('Environment changed since qualification')
            else:
                result = run_command([sys.executable, 'scripts/check_rental_host.py', '--directory', str(directory)],
                                     directory / 'preflight.log', min(600, runtime - 10))
                if not result['passed']:
                    raise RuntimeError('Host preflight failed: ' + str(result['failure']))
            report = read(directory / 'preflight.json')
            journal['preflight_sha256'] = sha(directory / 'preflight.json')
            allowed = ['qualification', 'performance', 'trace'][:['qualification', 'performance', 'trace'].index(through) + 1]
            for step in pipeline['steps']:
                if step['phase'] not in allowed:
                    break
                if step['name'] in {s['name'] for s in journal['steps']}:
                    continue
                remaining = journal['started_unix'] + runtime - time.time() - 10
                if remaining <= 0:
                    raise TimeoutError('Session budget exhausted')
                validate_hashes(ROOT, report['source_sha256'])
                host_budget(config, socket.gethostname(), [r[1] for r in gpu_rows()])
                journal['status'] = step['name']; write(directory / 'session.json', journal)
                result = run_command(step['command'], directory / f'{step["name"]}.log', min(step['timeout_s'], remaining))
                journal['steps'].append(dict(name=step['name'], **result))
                write(directory / 'session.json', journal)
                if not result['passed']:
                    raise RuntimeError(f'{step["name"]}: {result["failure"]}')
                gpu_rows()
            journal.update(passed=True, status='completed', completed_through=through)
        except BaseException as exc:
            journal.update(passed=False, status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            journal['elapsed_s'] = time.time() - journal['started_unix']
            journal['estimated_compute_cost'] = journal['elapsed_s'] * config['hourly_rate_all_gpus'] / 3600
            journal['cloud_billing_stopped'] = False
            try:
                journal['gpu_after'] = gpu_rows()
            except Exception as exc:
                journal.update(passed=False, cleanup_error=str(exc))
            write(directory / 'session.json', journal)
        return journal
