"""Exercise actual retry bindings and reject stale gates or occupied hardware."""
import json
from pathlib import Path

import pytest

from scripts import run_i2_precision as original
from scripts import run_i2_precision_followup as runner
from serving.i2_workloads import digest


@pytest.fixture
def gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(original, 'ROOT', tmp_path)
    config = dict(qualification_directory='q', lifecycle_directory='failed_default',
                  model_path='model', modes=['bf16_control', 'mixed'])
    bindings = dict(qualification_directory='q', lifecycle_directory='retry')
    for name in ('q', 'retry', 'scripts', 'model'):
        (tmp_path/name).mkdir()
    for name in ('audit_i2_precision.py', 'audit_i2_precision_shutdown.py'):
        (tmp_path/'scripts'/name).write_text('# frozen auditor\n')
    (tmp_path/'model/weights').write_text('fixed checkpoint')
    auditor = digest(tmp_path/'scripts/audit_i2_precision.py')
    for name, positive in (('q', 'qualification_passed'), ('retry', 'HTTP_lifecycle_passed')):
        p = tmp_path/name
        original.write(p/'workload.json', {'input_ids': [1, 2, 3]})
        original.write(p/'provenance.json', dict(config=config, source_sha256={},
            workload_sha256=digest(p/'workload.json'),
            model_sha256={'weights': digest(tmp_path/'model/weights')}))
        original.write(p/'audit.json', {positive: True, 'auditor_sha256': auditor,
            'provenance_sha256': digest(p/'provenance.json')})
    p = tmp_path/'retry'
    meta = original.read(p/'provenance.json')
    meta['qualification_audit_sha256'] = digest(tmp_path/'q/audit.json')
    original.write(p/'provenance.json', meta)
    audit = original.read(p/'audit.json')
    audit['provenance_sha256'] = digest(p/'provenance.json')
    original.write(p/'audit.json', audit)
    original.write(p/'shutdown_audit.json', dict(passed=True,
        HTTP_audit_sha256=digest(p/'audit.json'),
        source_sha256=digest(tmp_path/'scripts/audit_i2_precision_shutdown.py')))
    return tmp_path, config, bindings


def test_explicit_retry_binding_overrides_failed_default(gates):
    root, config, bindings = gates
    assert not (root/config['lifecycle_directory']).exists()
    gate = runner.prerequisites(config, bindings)
    assert gate['performance_allowed']
    assert gate['lifecycle_directory'] == 'retry'
    assert gate['lifecycle_audit_sha256'] == digest(root/'retry/audit.json')


@pytest.mark.parametrize('name,field', [('q/audit.json', 'qualification_passed'),
    ('retry/audit.json', 'HTTP_lifecycle_passed'), ('retry/shutdown_audit.json', 'passed')])
def test_actual_negative_gate_rejects_performance(gates, name, field):
    root, config, bindings = gates
    value = original.read(root/name)
    value[field] = False
    original.write(root/name, value)
    with pytest.raises(ValueError, match='gates required'):
        runner.prerequisites(config, bindings)


def test_mutated_checkpoint_rejects_stale_numerical_gate(gates):
    root, config, bindings = gates
    (root/'model/weights').write_text('different checkpoint')
    with pytest.raises(ValueError, match='checkpoint changed'):
        runner.prerequisites(config, bindings)


def test_changed_HTTP_audit_rejects_stale_shutdown_gate(gates):
    root, config, bindings = gates
    value = original.read(root/'retry/audit.json')
    value['unbound_change'] = True
    original.write(root/'retry/audit.json', value)
    with pytest.raises(ValueError, match='Gate binding changed'):
        runner.prerequisites(config, bindings)


def test_changed_shutdown_auditor_rejects_gate(gates):
    root, config, bindings = gates
    (root/'scripts/audit_i2_precision_shutdown.py').write_text('# modified\n')
    with pytest.raises(ValueError, match='Shutdown auditor changed'):
        runner.prerequisites(config, bindings)


@pytest.mark.parametrize('memory,allowed', [('0 MiB\n', True), ('92 MiB\n', False)])
def test_exclusive_hardware_gate(monkeypatch, memory, allowed):
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: memory)
    if allowed:
        assert runner.require_idle_gpu() == '0 MiB'
    else:
        with pytest.raises(RuntimeError, match='Exclusive GPU required'):
            runner.require_idle_gpu()
