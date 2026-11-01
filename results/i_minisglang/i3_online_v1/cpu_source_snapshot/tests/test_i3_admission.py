"""Reject stale or negative actual online admission gates before spending GPU time."""
import shutil

import pytest

from scripts import run_i3_formal as formal
from scripts import run_i3_online as base
from serving import i3_audit
from serving.i3_workloads import read,write,schedules
from serving.i2_workloads import digest,token_hash


@pytest.fixture
def gates(tmp_path,monkeypatch):
    for module in (formal,base,i3_audit):monkeypatch.setattr(module,'ROOT',tmp_path)
    for name in ('q','p','scripts','model'):(tmp_path/name).mkdir()
    for name in ('audit_i3_pilot.py','audit_i3_qualification.py'):(tmp_path/'scripts'/name).write_text('# frozen\n')
    (tmp_path/'model/weights').write_text('checkpoint')
    (tmp_path/'policy.py').write_text('# compute\n')
    config=dict(model_path='model',variants=dict(radix={},graph124={}),seed=2,
        rate_rule=dict(low_min_capacity_multiplier=.25,near_min_capacity_multiplier=.9,overload_max_capacity_multiplier=1.5),
        latency_objectives_s=dict(first_text=1,e2e=8,amortized_tpot=.1))
    workload=dict(requests=[dict(request_id='a'),dict(request_id='b')])
    for name in ('q','p'):
        directory=tmp_path/name;write(directory/'workload.json',workload)
        snap=directory/'source_snapshot';snap.mkdir();shutil.copy2(tmp_path/'policy.py',snap/'policy.py')
        write(directory/'provenance.json',dict(config=config,config_sha256=token_hash(config),
            workload_sha256=digest(directory/'workload.json'),source_sha256={'policy.py':digest(tmp_path/'policy.py')},
            model_sha256={'weights':digest(tmp_path/'model/weights')}))
    write(tmp_path/'q/audit.json',dict(passed=True,auditor_sha256=digest(tmp_path/'scripts/audit_i3_qualification.py'),
        provenance_sha256=digest(tmp_path/'q/provenance.json')))
    capacities=dict(radix=2,graph124=8)
    for variant,value in capacities.items():write(tmp_path/f'p/pilot_{variant}/capacity.json',dict(requests_per_s=value))
    write(tmp_path/'p/schedule.json',schedules(workload,capacities,config))
    write(tmp_path/'p/audit.json',dict(passed=True,auditor_sha256=digest(tmp_path/'scripts/audit_i3_pilot.py'),
        provenance_sha256=digest(tmp_path/'p/provenance.json'),schedule_sha256=digest(tmp_path/'p/schedule.json')))
    return tmp_path,config


def test_positive_explicit_bindings_include_both_independent_audits(gates):
    root,config=gates;g=formal.gates(root/'q',root/'p',config)
    assert g['performance_allowed'] and g['qualification_directory']=='q' and g['pilot_directory']=='p'
    assert g['qualification_audit_sha256']==digest(root/'q/audit.json')
    assert g['pilot_audit_sha256']==digest(root/'p/audit.json')


@pytest.mark.parametrize('stage',['q','p'])
def test_negative_independent_gate_rejects_formal(gates,stage):
    root,config=gates;p=root/stage/'audit.json';a=read(p);a['passed']=False;write(p,a)
    with pytest.raises(ValueError):formal.gates(root/'q',root/'p',config)


def test_mutated_arrival_table_is_rejected(gates):
    root,config=gates;p=root/'p/schedule.json';a=read(p);a['traces']['low'][1]['arrival_s']+=1;write(p,a)
    with pytest.raises(ValueError,match='schedule changed'):formal.gates(root/'q',root/'p',config)


def test_mutated_checkpoint_is_rejected(gates):
    root,config=gates;(root/'model/weights').write_text('changed')
    with pytest.raises(ValueError,match='sources changed'):formal.gates(root/'q',root/'p',config)


def test_mutated_compute_source_is_rejected(gates):
    root,config=gates;(root/'policy.py').write_text('# changed')
    with pytest.raises(ValueError,match='sources changed'):formal.gates(root/'q',root/'p',config)


@pytest.mark.parametrize('name',['audit_i3_pilot.py','audit_i3_qualification.py'])
def test_stale_auditor_is_rejected(gates,name):
    root,config=gates;(root/'scripts'/name).write_text('# changed')
    with pytest.raises(ValueError):formal.gates(root/'q',root/'p',config)


def test_changed_pilot_provenance_cannot_reuse_audit(gates):
    root,config=gates;p=root/'p/provenance.json';a=read(p);a['changed']=True;write(p,a)
    with pytest.raises(ValueError,match='pilot binding changed'):formal.gates(root/'q',root/'p',config)
