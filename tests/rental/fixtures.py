"""Explicit synthetic report fixtures, never multigpu experiment evidence."""
from copy import deepcopy
from pathlib import Path

from rental.common import ROOT, sha, write
from rental.session import plan
from serving.tp_preparation import load_plan, jobs
from scripts.prepare_f_training import jobs as f_jobs


def create(directory):
    directory.mkdir(parents=True)
    source = {'model.py': sha(ROOT / 'model.py')}
    p = plan(directory); write(directory / 'plan.json', p)
    write(directory / 'host_contract.json', {'synthetic_fixture': True})
    write(directory / 'preflight.json', dict(source_sha256=source, synthetic_fixture=True))
    journal = []
    for step in p['steps']:
        path = directory / f'{step["name"]}.log'; path.write_text('synthetic test fixture\n')
        journal.append(dict(name=step['name'], passed=True, log_sha256=sha(path)))
    write(directory / 'session.json', dict(passed=True, status='completed', completed_through='trace', steps=journal,
        plan_sha256=sha(directory / 'plan.json'), preflight_sha256=sha(directory / 'preflight.json'),
        host_contract_sha256=sha(directory / 'host_contract.json')))
    tp = directory / 'tp'; c = load_plan()
    workload = dict(cases=[dict(case_id='test_case', waves={phase: [dict(request_id=phase, continuation_ids=[1, 2])]
                                                        for phase in ('cold', 'warm')})])
    write(tp / 'workload.json', workload)
    processes = []
    events = dict(traceEvents=[dict(ph='X', cat='kernel', name='ncclAllReduce', ts=0, dur=3),
                              dict(ph='X', cat='kernel', name='gemm', ts=2, dur=3)])
    for stage in ('benchmark', 'trace'):
        for job in jobs(c, stage):
            log = tp / 'logs' / f'{job["label"]}.log'
            log.parent.mkdir(parents=True, exist_ok=True); log.write_text('synthetic fixture\n')
            processes.append(dict(label=job['label'], returncode=0, escalated=[], log_sha256=sha(log)))
            for rank in range(job['world']):
                waves = [dict(repeat=repeat, case_id='test_case', phase=phase,
                              released=True, measured=stage == 'benchmark' and repeat >= 3,
                              seconds=2 / job['world'] + rank * .01,
                              requests=[dict(request_id=phase, output_tokens=2, actual_tokens=[1, 2])])
                         for repeat in range(8 if stage == 'benchmark' else 1) for phase in ('cold', 'warm')]
                report = dict(passed=True, rank=rank, mode=stage, **job, waves=waves,
                              workload_sha256=sha(tp / 'workload.json'), config_sha256=sha(ROOT / 'config/delivery/tp_matrix.json'),
                              shutdown_s=.1, peak_allocated_bytes=1000, peak_reserved_bytes=2000)
                if stage == 'trace':
                    trace = tp / 'traces' / f'{job["label"]}_rank{rank}.json'; write(trace, events)
                    report['trace_sha256'] = sha(trace)
                write(tp / 'raw' / job['label'] / f'rank{rank}.json', report)
    write(tp / 'processes.json', processes)
    f = directory / 'f'
    write(f / 'training/cuda1_full/job.json', dict(host='fixture'))
    for job in f_jobs(benchmark=True):
        path = f / 'performance' / job['label']
        write(path / 'job.json', dict(**job, returncode=0, timed_out=False, source_sha256=source, host='fixture'))
        (path / 'worker.log').write_text('synthetic fixture\n')
        for rank in range(job['world']):
            write(path / f'rank{rank}.json', dict(rank=rank, world=job['world'], tiny=False, dtype='bfloat16',
                  accumulation=4, global_tokens_per_update=16384, peak_allocated=1024, window_sha256='fixture_windows',
                  entries=[dict(mode=job['mode'], warmup=20, update_seconds=[1 / job['world']] * 100,
                                last_loss=3, last_global_grad_norm=1)]))
    write(f / 'trace/job.json', dict(source_sha256=source))
    reports = {}
    for rank in (0, 1):
        write(f / f'trace/trace_rank{rank}.json', events)
        path = f / f'trace/rank{rank}.json'; write(path, {'synthetic_fixture': True})
        reports[path.name] = sha(path)
    write(f / 'trace/audit.json', dict(passed=True, world=2, device='cuda', cross_gpu_accepted=True,
          job_sha256=sha(f / 'trace/job.json'), raw_sha256={}, report_sha256=reports, traffic=[]))
    return directory
