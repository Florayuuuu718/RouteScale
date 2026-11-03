"""Reopen local delivery evidence; never promote TP preparation to multi-GPU acceptance.

This final gate does not launch servers. Large tensors and build archives remain
in explicitly selected local directories and are verified by streaming SHA256.
Container PID liveness was audited inside its PID namespace, before archival.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hashes_match(directory, hashes):
    return all((directory / name).is_file() and sha(directory / name) == value
               for name, value in hashes.items())


def accept(directory, tp_runtime, bundle_directory):
    checks = {}
    counts = {}
    for group, names in {
        'mini': ['mini_healthy', 'mini_worker_fault', 'mini_shutdown_active'],
        'sglang_native': ['sglang_native_healthy', 'sglang_native_worker_fault', 'sglang_native_shutdown_active'],
        'sglang_container': ['sglang_container_healthy'],
    }.items():
        terminal = Counter()
        audit_checks = 0
        for name in names:
            d = directory / name
            audit = read(d / 'audit.json')
            run = read(d / 'run.json')
            rows = read(d / 'requests.json')
            checks[name + '/audit'] = audit['passed'] and all(audit['checks'].values())
            checks[name + '/evidence'] = hashes_match(d, audit['evidence_sha256'])
            checks[name + '/requests'] = len(rows) == audit['requests'] and len({r['request_id'] for r in rows}) == len(rows)
            checks[name + '/actual_source'] = hashes_match(ROOT, run.get('source_sha256', {}))
            checks[name + '/run'] = run['passed'] and bool(run['root_commit'])
            audit_checks += len(audit['checks'])
            terminal.update(r['state'] for r in rows)
        counts[group] = dict(requests=sum(terminal.values()), terminal=dict(terminal), independent_checks=audit_checks)
    checks['mini_scope'] = counts['mini']['requests'] == 21 and counts['mini']['independent_checks'] == 131
    checks['native_scope'] = counts['sglang_native']['requests'] == 75 and counts['sglang_native']['independent_checks'] == 705

    native = directory / 'sglang_native_healthy'
    exported = read(native / 'export.json')
    checks['actual_monitoring_export'] = (exported['passed'] and exported['samples'] >= 2
        and exported['audit_sha256'] == sha(native / 'audit.json')
        and hashes_match(native, exported['artifacts_sha256'])
        and not exported['performance_ranking_claimed'])

    tp = read(directory / 'tp/local004/local_audit.json')
    checks['TP_local_qualification'] = tp['passed'] and all(tp['checks'].values()) and tp['vectors'] == 640 and len(tp['qualified_labels']) == 10
    checks['TP_archived_evidence'] = hashes_match(directory / 'tp/local004', {
        n: h for n, h in tp['evidence_sha256'].items() if not n.endswith('.pt')})
    checks['TP_local_tensor_evidence'] = hashes_match(tp_runtime, tp['evidence_sha256'])
    checks['TP_current_source'] = hashes_match(ROOT, tp['source_sha256'])
    processes = [read(p) for p in (directory / 'tp/local004/logs').glob('*.process.json')]
    checks['TP_12_processes'] = len(processes) == 12 and all(
        p['returncode'] == 0 and p['status'] == 'pass' and not p['escalated'] for p in processes)
    completed = read(directory / 'tp/local004/processes.json')
    checks['TP_all_process_GPU_cleanup'] = len(completed) == 12 and all(
        p['returncode'] == 0 and not p['escalated'] and p['gpu_after'].strip() == '0, 0' for p in completed)
    plan = read(directory / 'tp/plan.json')
    checks['TP_future_matrix'] = (len(plan['qualification']) == 30 and len(plan['benchmark']) == 90
        and not plan['multigpu_accepted'] and not plan['paid_resources_authorized']
        and not tp['multigpu_accepted'] and not tp['paid_resources_used'])

    bundle = read(directory / 'sglang_framework_bundle.json')
    container = read(directory / 'sglang_container_result.json')
    stop = read(directory / 'sglang_docker_stop_result.json')
    checks['offline_archives'] = hashes_match(bundle_directory, bundle['archives_sha256'])
    checks['locked_206_dependencies'] = (bundle['doctor']['passed']
        and len(bundle['doctor']['environment']['packages']) == 206
        and bundle['doctor']['environment']['packages']['sglang'] == '0.5.20'
        and bundle['requirements_sha256'] == sha(ROOT / 'deploy/local/sglang/requirements.lock'))
    container_doctor = read(directory / 'sglang_container_healthy/status.json')['doctor']
    checks['container_exact_environment'] = (container_doctor['environment'] == bundle['doctor']['environment']
        and container_doctor['upstream_source_sha256'] == bundle['doctor']['upstream_source_sha256']
        and container_doctor['toolchain']['math_header_sha256'] == bundle['container_toolchain_header_sha256'])
    checks['container_source'] = (bundle['clean_source_checkout'] and not bundle['weights_embedded']
        and container['source_commit'] == stop['source_commit'] == bundle['source_commit']
        and hashes_match(ROOT, bundle['source_sha256']))
    checks['container_healthy'] = container['passed'] and container['returncode'] == 0 and container['GPU_end'] == '0 MiB'
    checks['container_image_identity'] = container['image_id'] == stop['image_id'] and container['image_id'].startswith('sha256:')
    files = read(directory / 'sglang_container_files_proof.json')
    checks['container_all_source_and_parent_layers'] = (files['passed'] and files['source_files'] == len(bundle['source_sha256'])
        and not files['mismatches'] and files['header_matches'] and files['parent_rootfs_prefix_verified']
        and files['source_commit'] == bundle['source_commit'] and files['image_id'] == container['image_id']
        and files['derived_parent_image_id'] == bundle['derived_base_image_id'])
    warm = read(directory / 'sglang_container_warmup/warmup.json')
    checks['declared_representative_warmup'] = (warm['passed'] and warm['requests'] == 28
        and warm['public_client_budget_s'] == 30 and warm['shutdown_budget_s'] == 20
        and warm['startup_budget_s'] == 600 and warm['internal_warmup_uses_remaining_startup_budget']
        and warm['native_status']['GPU_end'] == '0 MiB' and not warm['native_status']['forced_cleanup']
        and warm['warmup_source_sha256'] == sha(ROOT / 'scripts/warm_framework.py'))
    warm_rows = read(directory / 'sglang_container_warmup/requests.json')
    checks['warmup_28_native_terminal_proofs'] = len(warm_rows) == len({r['request_id'] for r in warm_rows}) == 28 and all(
        r['state'] == 'finished' and r['finish_reason'] == 'length'
        and sum(e['event'] == 'finished' for e in r['events']) == 1
        and any(e['event'] == 'framework_frame' and (e['native'].get('choices') or [{}])[0].get('finish_reason') == 'length'
                for e in r['events']) for r in warm_rows)
    checks['container_default_entrypoint_stop'] = (stop['passed'] and stop['actual_supervisor_PID1']
        and stop['readonly_model'] and stop['container_exit_code'] == 0
        and stop['docker_stop_returncode'] == 0 and stop['docker_stop_s'] < 30
        and stop['native_loads_at_stop']['loads'][0]['num_running_reqs'] >= 1
        and stop['native_status']['state'] == 'stopped' and not stop['native_status']['forced_cleanup']
        and stop['native_status']['shutdown_s'] <= 20.5 and stop['client_returncode'] == 0
        and stop['native_status']['GPU_end'] == stop['GPU_end'] == '0 MiB'
        and stop['client_rows'][0]['state'] == 'finished')
    row = stop['client_rows'][0]
    frames = [e['native'] for e in row['events'] if e['event'] == 'framework_frame']
    checks['container_stop_client_wire_proof'] = (len(stop['client_rows']) == 1 and bool(row['content'])
        and sum(e['event'] == 'finished' for e in row['events']) == 1
        and ''.join((f.get('choices') or [{}])[0].get('text') or '' for f in frames) == row['content']
        and any((f.get('choices') or [{}])[0].get('finish_reason') == 'length' for f in frames))
    probe = read(directory / 'sglang_windows_host_probe.json')
    checks['published_API_and_dashboard'] = (probe['passed'] and probe['ready_status'] == probe['dashboard_status'] == 200
        and probe['dashboard_has_cards'] and 'RouteScale-Qwen3-0.6B' in probe['model_ids'])

    historical = read(directory / 'historical.json')
    checks['1498_historical_JSON_unchanged'] = len(historical) == 1498 and hashes_match(ROOT, historical)
    model = read(ROOT / 'config/delivery/model.json')
    checks['ten_model_files'] = len(model['files']) == 10 and hashes_match(ROOT / 'runs/models/Qwen3-0.6B', model['files'])
    environment = read(directory / 'cpu_environment.json')
    cpu = (directory / 'cpu_tests.txt').read_text()
    test_count = re.search(r'Ran (\d+) tests', cpu)
    checks['CPU_tests'] = bool(test_count) and int(test_count[1]) >= 51 and cpu.rstrip().endswith('OK')
    checks['CPU_environment_without_Torch'] = environment['no_Torch_or_GPU_dependencies'] and 'torch' not in environment['packages']
    checks['CPU_static'] = read(directory / 'cpu_static.json')['passed']
    ci = read(directory / 'github_ci_source.json')
    checks['remote_source_CI'] = ci['head_sha'] == bundle['source_commit'] and ci['status'] == 'completed' and ci['conclusion'] == 'success'
    with (ROOT / 'runs/tmp/h2_runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'], text=True).strip()
        checks['GPU_idle_and_lock_free'] = gpu == '0 MiB'
    return dict(passed=all(checks.values()), local_delivery_complete=all(checks.values()), checks=checks,
        counts=counts, cpu_tests=int(test_count[1]) if test_count else 0, TP_local_vectors=tp['vectors'],
        TP2_4_accepted=False, paid_resources_used=False, native_HF_FP32_equivalence_claimed=False,
        cold_all_workload_shapes_accepted=False, warm_JIT_cache_used_in_full_container_scenario=True,
        steady_state_performance_ranking_claimed=False, GPU_end=gpu,
        evidence_sha256={str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
                         if p.is_file() and p.name != 'acceptance.json'},
        acceptor_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/delivery/local_v1')
    parser.add_argument('--tp-runtime', type=Path, required=True)
    parser.add_argument('--bundle-directory', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Recheck an existing acceptance without replacing it')
    args = parser.parse_args()
    target = args.directory / 'acceptance.json'
    if target.exists() and not args.check_only:
        raise ValueError('Preserve the final acceptance; use a new result scope')
    result = accept(args.directory, args.tp_runtime, args.bundle_directory)
    if args.check_only:
        previous = read(target)
        result['checks']['recorded_acceptance_evidence'] = hashes_match(args.directory, previous['evidence_sha256'])
        result['checks']['recorded_acceptor'] = previous['acceptor_sha256'] == sha(Path(__file__))
        result['passed'] = result['local_delivery_complete'] = all(result['checks'].values()) and previous['passed']
    else:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'counts', 'cpu_tests', 'GPU_end')}))
    if not result['passed']:
        print('Failed:', [k for k, v in result['checks'].items() if not v])
        raise SystemExit(1)
