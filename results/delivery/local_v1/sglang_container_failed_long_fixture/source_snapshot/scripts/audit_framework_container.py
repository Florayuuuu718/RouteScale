"""Reopen the explicit container queue-handshake scope with all original gates."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_framework import audit
from serving.delivery import read, write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    target = args.directory / 'audit.json'
    if target.exists():
        raise ValueError('Preserve the original audit')
    result = audit(args.directory)
    report = read(args.directory / 'run.json')
    catalogue = read(args.directory / 'request_catalogue.json')
    blockers = [r for r in catalogue if r['request_id'].startswith('queue_blocker_')]
    rows = read(args.directory / 'requests.json')
    queued = next(r for r in rows if r['request_id'] == 'queued_disconnect')
    result['checks']['declared_container_handshake_scope'] = report['diagnostic_scope'] == 'container_queue_handshake_v1'
    result['checks']['four_long_blockers'] = len(blockers) == 4 and all(r['payload']['max_tokens'] == 900 and r['timeout_s'] == 30 for r in blockers)
    workload = read(args.directory / 'source_snapshot/results/i_minisglang/i3_online_v2/workload002/workload.json')
    first = workload['requests'][0]
    result['checks']['long_fixture_context_and_prompt'] = len(first['input_ids']) + 900 <= 1024 and all(r['payload']['prompt'] == first['payload']['prompt'] for r in blockers)
    result['checks']['queue_observation_triggers_real_close'] = queued['reason'] == 'observed_queue_connection_close'
    result['passed'] = all(result['checks'].values())
    write(target, result)
    print('Container handshake audit', result['passed'], len(result['checks']), 'failed', [k for k, v in result['checks'].items() if not v])
    if not result['passed']:
        raise SystemExit(1)
