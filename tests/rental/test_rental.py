"""CPU acceptance of deployment integrity, process cleanup and report semantics."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rental.common import ROOT, read, relative, sha, write
from rental.deployment import verify_bundle
from rental.report import comparisons, export, measured_waves, trace_summary, training_record, summarize_tp, summarize_training
from rental.session import host_budget, plan, process_table, run_command, resume_state
from serving.tp_preparation import load_plan
from fixtures import create


class RentalTests(unittest.TestCase):
    def test_plan_covers_f_h2_i2_and_qualifies_before_timing(self):
        p = plan(Path('/tmp/session'))
        self.assertEqual(p['counts'], dict(tp_qualification=30, tp_performance=90, tp_trace=30,
            f_module_qualification=8, f_training_qualification=5, f_performance=9, f_trace=1))
        phases = [s['phase'] for s in p['steps']]
        self.assertEqual(phases, sorted(phases, key=['qualification', 'performance', 'trace'].index))
        for step in p['steps']:
            self.assertTrue((ROOT / step['command'][1]).is_file())

    def test_plan_only_cli_needs_neither_cuda_nor_target_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'plan'
            subprocess.run([sys.executable, 'scripts/run_rental.py', '--directory', str(target)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            self.assertFalse((target / 'session.json').exists())
            self.assertEqual(read(target / 'plan.json')['gpu_count'], 4)

    def contract(self):
        return dict(existing_host_confirmed=True, hostname='host', gpu_uuids=['a', 'b', 'c', 'd'],
                    max_runtime_s=3600, hourly_rate_all_gpus=10, max_cost=10, currency='CNY')

    def test_valid_host_budget(self):
        self.assertEqual(host_budget(self.contract(), 'host', ['a', 'b', 'c', 'd']), 3600)

    def test_missing_host_or_budget_rejected(self):
        with self.assertRaises(ValueError):
            host_budget(read(ROOT / 'deploy/rental/host.example.json'), 'host', ['a', 'b', 'c', 'd'])

    def test_wrong_host_and_reordered_gpu_rejected(self):
        for host, uuids in [('other', ['a', 'b', 'c', 'd']), ('host', ['b', 'a', 'c', 'd']), ('host', ['a'] * 4)]:
            with self.subTest(host=host, uuids=uuids), self.assertRaises(ValueError):
                host_budget(self.contract(), host, uuids)

    def test_nonfinite_or_over_budget_rejected(self):
        for key, value in [('max_runtime_s', math.nan), ('max_cost', math.inf), ('hourly_rate_all_gpus', 0),
                           ('max_runtime_s', 3601), ('max_runtime_s', True)]:
            c = self.contract(); c[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                host_budget(c, 'host', ['a', 'b', 'c', 'd'])

    def test_manifest_paths_cannot_escape_bundle(self):
        for name in ('../secret', '/etc/passwd', 'assets/../../secret', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(name)

    def test_bundle_tamper_extra_file_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'asset').write_bytes(b'pinned')
            write(root / 'bundle.json', dict(files={'asset': sha(root / 'asset')}))
            verify_bundle(root)
            (root / 'asset').write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_bundle(root)
            (root / 'asset').write_bytes(b'pinned')
            (root / 'extra').write_bytes(b'unlisted')
            with self.assertRaises(ValueError): verify_bundle(root)
            (root / 'extra').unlink(); (root / 'extra').symlink_to(root / 'asset')
            with self.assertRaises(ValueError): verify_bundle(root)

    def timing_fixture(self):
        config = load_plan()
        job = dict(stage='benchmark', world=2, label='test', framework='mini', variant='radix_full')
        request = dict(request_id='r', continuation_ids=[1, 2])
        workload = dict(cases=[dict(case_id='case', waves={p: [request] for p in ('cold', 'warm')})])
        waves = [dict(repeat=i, case_id='case', phase=p, released=True, seconds=1,
                      measured=i >= config['warmup_repeats'], requests=[dict(request_id='r', output_tokens=2)])
                 for i in range(8) for p in ('cold', 'warm')]
        return dict(job, passed=True, mode='benchmark', waves=waves), job, workload, config

    def test_warmup_excluded_and_complete_work_accounted(self):
        args = self.timing_fixture()
        self.assertEqual(len(measured_waves(*args)), 10)

    def test_tp_duplicates_partial_work_and_warmup_contamination_rejected(self):
        for change in ('duplicate', 'missing', 'tokens', 'warmup', 'released', 'nan'):
            r, j, w, c = self.timing_fixture()
            if change == 'duplicate': r['waves'].append(deepcopy(r['waves'][0]))
            if change == 'missing': r['waves'].pop()
            if change == 'tokens': r['waves'][0]['requests'][0]['output_tokens'] = 1
            if change == 'warmup': r['waves'][0]['measured'] = True
            if change == 'released': r['waves'][0]['released'] = False
            if change == 'nan': r['waves'][0]['seconds'] = math.nan
            with self.subTest(change=change), self.assertRaises(ValueError): measured_waves(r, j, w, c)

    def test_training_checks_complete_tokens_steps_precision_and_finite_state(self):
        c = read(ROOT / 'config/f_ep_rental_requirements.json')
        j = dict(world=2, mode='ep')
        r = dict(rank=0, world=2, tiny=False, dtype='bfloat16', accumulation=4, global_tokens_per_update=16384,
                 peak_allocated=1024, entries=[dict(mode='ep', warmup=20, update_seconds=[1.] * 100,
                                                   last_loss=3., last_global_grad_norm=.5)])
        self.assertEqual(len(training_record(r, j, 0, c)), 100)
        for field, value in [('global_tokens_per_update', 8192), ('world', 1), ('tiny', True), ('rank', 1)]:
            bad = dict(r, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError): training_record(bad, j, 0, c)
        bad = deepcopy(r); bad['entries'][0]['last_loss'] = math.nan
        with self.assertRaises(ValueError): training_record(bad, j, 0, c)
        bad = deepcopy(r); bad['entries'][0]['update_seconds'].pop()
        with self.assertRaises(ValueError): training_record(bad, j, 0, c)

    def test_independent_process_ratios_not_pooled_wave_speedup(self):
        rows = [dict(mode=mode, run=run, speed=speed, peak_rank_allocated=1)
                for mode, speeds in [('base', [1, 10, 100]), ('ep', [2, 20, 50])]
                for run, speed in enumerate(speeds, 1)]
        stats = comparisons(rows, ['mode'], 'speed',
                            lambda row, rows: {r['run']: r for r in rows if r['mode'] == 'base'})
        self.assertEqual(stats[1]['paired_speedup']['median'], 2)
        self.assertEqual(stats[1]['performance']['n'], 3)
        with self.assertRaises(ValueError):
            comparisons(rows[:-1], ['mode'], 'speed', lambda row, rows: {r['run']: r for r in rows if r['mode'] == 'base'})

    def test_trace_intervals_do_not_double_count_overlap_or_cpu_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.json'
            write(path, dict(traceEvents=[dict(ph='X', cat='kernel', name=name, ts=start, dur=duration)
                                         for name, start, duration in [('ncclAllReduce', 0, 10), ('ncclSend', 5, 10),
                                                                       ('gemm', 10, 10)]] +
                                        [dict(ph='X', cat='cpu_op', name='nccl', ts=0, dur=1000)]))
            result = trace_summary(path)
            self.assertEqual((result['nccl_union_us'], result['other_kernel_union_us'], result['overlap_us']), (15, 10, 5))
            write(path, dict(traceEvents=[]))
            with self.assertRaises(ValueError): trace_summary(path)

    def test_incomplete_session_preserved_without_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / 'session'
            write(directory / 'session.json', dict(passed=False, status='failed', steps=[dict(name='f_module', passed=False)]))
            write(directory / 'plan.json', dict(steps=[{}] * 11))
            result = export(directory, root / 'report')
            self.assertFalse(result['multigpu_accepted'])
            self.assertNotIn('tp', result)
            self.assertEqual(len(result['recorded_steps']), 1)

    def test_real_subprocess_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passed = run_command([sys.executable, '-c', 'print("ok")'], root / 'pass.log', 2)
            failed = run_command([sys.executable, '-c', 'raise SystemExit(7)'], root / 'fail.log', 2)
            self.assertTrue(passed['passed']); self.assertFalse(failed['passed'])
            self.assertEqual(failed['returncode'], 7)

    def test_real_timeout_cleans_detached_grandchild_keeps_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pidfile = root / 'child.pid'
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
            try:
                code = ('import subprocess,sys,time; from pathlib import Path; '
                        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); '
                        f'Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)')
                result = run_command([sys.executable, '-c', code], root / 'timeout.log', .5)
                self.assertFalse(result['passed']); self.assertEqual(result['failure'], 'timeout')
                self.assertFalse(result['descendants_remaining'])
                table = process_table(); child = int(pidfile.read_text())
                self.assertTrue(child not in table or table[child][2] == 'Z')
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate(); unrelated.wait(timeout=3)

    def test_immediate_parent_exit_cannot_leave_detached_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pidfile = root / 'orphan.pid'
            code = ('import subprocess,sys; from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); '
                    f'Path({str(pidfile)!r}).write_text(str(p.pid))')
            result = run_command([sys.executable, '-c', code], root / 'orphan.log', 3)
            self.assertFalse(result['passed'])
            self.assertTrue(result['cleanup_required'])
            self.assertNotIn(int(pidfile.read_text()), process_table())

    def test_full_synthetic_matrix_exports_and_uses_slowest_rank_without_multiplying_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = create(Path(tmp) / 'session')
            # Only numerical tensor gates are substituted. Real tensor audit
            # correctness remains covered by the already qualified TP/F suites.
            with patch('rental.report.numerical_gate'), patch('rental.report.performance_gate'):
                result = export(directory, Path(tmp) / 'export')
            self.assertTrue(result['passed'], result.get('error'))
            self.assertEqual(len(result['tp']), 60)
            self.assertEqual(len(result['training']), 3)
            self.assertEqual(len(result['traces']), 72)
            row = next(r for r in result['tp'] if r['world'] == 2)
            self.assertAlmostEqual(row['performance']['median'], 2 / 1.01)
            self.assertTrue((Path(tmp) / 'export/tp_processes.csv').is_file())

    def test_synthetic_ranks_windows_missing_processes_and_changed_logs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = create(Path(tmp) / 'session')
            target = directory / 'f/performance/ep_run2/rank1.json'
            original = read(target); bad = deepcopy(original); bad['window_sha256'] = 'changed'
            write(target, bad)
            with patch('rental.report.performance_gate'), self.assertRaisesRegex(ValueError, 'windows'):
                summarize_training(directory / 'f')
            bad = deepcopy(original); bad['entries'][0]['update_seconds'][0] += .1; write(target, bad)
            with patch('rental.report.performance_gate'), self.assertRaisesRegex(ValueError, 'rank-max'):
                summarize_training(directory / 'f')
            write(target, original)
            journal = read(directory / 'tp/processes.json'); journal.pop(0)
            write(directory / 'tp/processes.json', journal)
            with patch('rental.report.numerical_gate'), self.assertRaisesRegex(ValueError, 'process'):
                summarize_tp(directory / 'tp')
            (directory / 'tp_qualification.log').write_text('changed')
            result = export(directory, Path(tmp) / 'rejected')
            self.assertFalse(result['passed']); self.assertIn('Stage log changed', result['error'])

    def test_resume_only_unchanged_successful_phase_with_original_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = create(Path(tmp) / 'session'); config = read(directory / 'host_contract.json')
            self.assertTrue(resume_state(directory, config)['passed'])
            with self.assertRaisesRegex(ValueError, 'budget'):
                resume_state(directory, dict(config, max_cost=9999))
            journal = read(directory / 'session.json'); journal['passed'] = False
            write(directory / 'session.json', journal)
            with self.assertRaisesRegex(ValueError, 'retries'):
                resume_state(directory, config)


if __name__ == '__main__':
    unittest.main()
