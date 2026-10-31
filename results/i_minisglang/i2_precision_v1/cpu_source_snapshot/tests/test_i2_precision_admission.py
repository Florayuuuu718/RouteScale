"""Reject unqualified/changed precision scopes before formal timing."""
import copy
import json
from pathlib import Path

import pytest

from scripts import run_i2_precision as runner


@pytest.fixture
def gates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    config = dict(qualification_directory='q', lifecycle_directory='l', modes=['bf16_control','mixed'],
        numerics=dict(atol=.125,rtol=.02,max_relative_rmse=.005))
    (tmp_path/'scripts').mkdir()
    auditor = tmp_path/'scripts/audit_i2_precision.py'
    auditor.write_text('# frozen gate auditor\n')
    q, l = tmp_path/'q', tmp_path/'l'
    q.mkdir(); l.mkdir()
    runner.write(q/'provenance.json', dict(config=config,source_sha256={}))
    runner.write(q/'audit.json', dict(qualification_passed=True,auditor_sha256=runner.digest(auditor)))
    runner.write(l/'provenance.json', dict(config=config,source_sha256={},qualification_audit_sha256=runner.digest(q/'audit.json')))
    runner.write(l/'audit.json', dict(HTTP_lifecycle_passed=True,auditor_sha256=runner.digest(auditor)))
    return config, q, l, auditor


def test_both_positive_gates_bind_actual_audit_hashes(gates):
    config,q,l,_ = gates
    result = runner.prerequisites(config)
    assert result['performance_allowed']
    assert result['qualification_audit_sha256'] == runner.digest(q/'audit.json')
    assert result['lifecycle_audit_sha256'] == runner.digest(l/'audit.json')


@pytest.mark.parametrize('stage,field', [('q','qualification_passed'),('l','HTTP_lifecycle_passed')])
def test_failed_numerics_or_HTTP_reject_timing(gates, stage, field):
    config,q,l,_ = gates
    path = (q if stage == 'q' else l)/'audit.json'
    value = runner.read(path); value[field] = False; runner.write(path,value)
    with pytest.raises(ValueError,match='numerical and HTTP'):
        runner.prerequisites(config)


def test_changed_precision_threshold_scope_rejects_timing(gates):
    config,_,_,_ = gates
    modified = copy.deepcopy(config)
    modified['numerics']['max_relative_rmse'] = .01
    with pytest.raises(ValueError,match='config changed'):
        runner.prerequisites(modified)


def test_changed_auditor_rejects_stale_gates(gates):
    config,_,_,auditor = gates
    auditor.write_text('# changed auditor\n')
    with pytest.raises(ValueError,match='auditor changed'):
        runner.prerequisites(config)
