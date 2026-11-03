"""Prepare declared SGLang request shapes before the unchanged delivery checks.

Own one temporary server, preserve all warmup offers, then release the GPU.
The normal 30s client and 20s shutdown budgets still apply. This preparation
does not claim that arbitrary new shapes are compiled by native /ready.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.delivery import read, sha, write
from serving.framework_client import CompletionClient
from serving.online_client import OnlineRequest
from serving.sglang_framework import load

WORKLOAD = ROOT / 'results/i_minisglang/i3_online_v2/workload002/workload.json'


def plan(workload, model):
    candidates = [q for q in workload['requests'] if len(q['input_ids']) + 256 <= 1024]
    if not candidates:
        raise ValueError('No bounded representative prompt')
    prompts = [min(candidates, key=lambda q: len(q['input_ids'])), max(candidates, key=lambda q: len(q['input_ids']))]
    return [[OnlineRequest(f'warm_shape{s}_o{tokens}_{i}',
                dict(q['payload'], model=model, max_tokens=tokens, ignore_eos=True), timeout_s=30)
             for i in range(4)] for s, q in enumerate(prompts) for tokens in (32, 64, 256)]


async def exercise(c, out):
    catalogue = []
    rows = []
    def sink(event):
        with (out / 'client_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
    for batch in plan(read(WORKLOAD), c['server_args']['served_model_name']):
        catalogue.extend(dict(request_id=r.request_id, payload=r.payload, timeout_s=r.timeout_s) for r in batch)
        write(out / 'request_catalogue.json', catalogue)
        result = await CompletionClient(f'http://127.0.0.1:{c["port"]}', concurrency=4, sink=sink).run(batch)
        rows.extend(result)
        write(out / 'requests.json', rows)
        if not all(r['state'] == 'finished' and r['finish_reason'] == 'length' for r in result):
            raise RuntimeError('Representative warmup failed at the unchanged client budget')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory.resolve()
    out.mkdir(parents=True, exist_ok=False)
    c = load()
    result = dict(passed=False, workload_sha256=sha(WORKLOAD), warmup_source_sha256=sha(Path(__file__)),
        shapes='shortest/longest eligible I3 prompt; concurrency4; outputs32/64/256',
        public_client_budget_s=30, shutdown_budget_s=c['shutdown_s'], arbitrary_cold_shapes_qualified=False)
    process = None
    try:
        with (out / 'supervisor.txt').open('x') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/framework_service.py'), 'run', '--run-dir', str(out / 'runtime')],
                cwd=ROOT, env=dict(os.environ, ROUTESCALE_CLIENT_LEDGER=str(out / 'requests.json')), stdout=log, stderr=subprocess.STDOUT)
            end = time.monotonic() + c['startup_s'] + 15
            while time.monotonic() < end:
                if process.poll() is not None:
                    raise RuntimeError('Preparation server exited before readiness')
                status = out / 'runtime/status.json'
                if status.exists() and read(status)['state'] == 'ready':
                    break
                time.sleep(.2)
            else:
                raise TimeoutError('Preparation startup budget')
            rows = asyncio.run(exercise(c, out))
            result.update(requests=len(rows), terminal_counts={'finished': len(rows)})
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=c['shutdown_s'] + 15)
            status = read(out / 'runtime/status.json')
            result['passed'] = (len(rows) == 24 and process.returncode == 0 and status['state'] == 'stopped'
                and not status['forced_cleanup'] and status['GPU_end'] == '0 MiB')
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=c['shutdown_s'] + 15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result['forced_supervisor_cleanup'] = True
        if (out / 'runtime/status.json').exists():
            result['native_status'] = read(out / 'runtime/status.json')
        write(out / 'warmup.json', result)
    print(json.dumps(dict(passed=result['passed'], requests=result.get('requests'), error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
