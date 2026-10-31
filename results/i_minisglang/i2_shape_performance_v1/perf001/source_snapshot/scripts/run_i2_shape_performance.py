"""Admission-controlled 21-process matrix for the explicit qualified I2 scope."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest,token_hash


def prerequisites(spec,config,workload):
    q=ROOT/spec['qualification_directory'];l=ROOT/spec['lifecycle_directory']
    qa=json.loads((q/'audit.json').read_text());qm=json.loads((q/'provenance.json').read_text())
    lm=json.loads((l/'provenance.json').read_text());ls=json.loads((l/'summary.json').read_text())
    la=json.loads((l/'independent_audit.json').read_text())
    if not qa['integrity_passed'] or not qa['qualification_passed'] or not ls['HTTP_lifecycle_passed']:raise ValueError('Both numerical and HTTP gates must pass')
    if not la['passed'] or la['provenance_sha256']!=digest(l/'provenance.json'):raise ValueError('Independent HTTP audit required')
    if qa['source_sha256']!=digest(ROOT/'scripts/audit_i2_shape_qualification.py') or la['source_sha256']!=digest(ROOT/'scripts/audit_i2_shape_lifecycle.py'):raise ValueError('Gate auditor source changed')
    if qa['reference_scope']!=spec['reference_scope'] or qm['reference_scope']!=spec['reference_scope'] or config['numerics']['reference']!='independent_shape_controlled_naive_BF16':raise ValueError('Reference scope mismatch')
    if lm['qualification_sha256']!=digest(q/'audit.json'):raise ValueError('HTTP scope is not bound to this numerical qualification')
    for n in ('serving/i2_runtime.py','serving/i2_worker.py','serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py'):
        if digest(ROOT/n)!=qm['source_sha256'][n]:raise ValueError('Qualified runtime source changed')
    if not all(digest(ROOT/n)==sha for n,sha in lm['source_sha256'].items()):raise ValueError('Qualified HTTP sources changed')
    for key in ('model_path','engine','variants','prompt_tokens','output_tokens','shared_prefix_tokens','batch_sizes','seed'):
        if config[key]!=qm['config'][key]:raise ValueError(f'Compute scope changed: {key}')
    for key in ('atol','rtol','max_relative_rmse','allow_bf16_reduced_precision_reduction'):
        if config['numerics'][key]!=qm['config']['numerics'][key]:raise ValueError(f'Numerical policy changed: {key}')
    control=next(s for s in json.loads((q/'workload.json').read_text())['suites'] if s['name']=='control512')['workload']
    if workload!=control:raise ValueError('Performance input differs from qualified subset')
    for n,sha in qm['model_sha256'].items():
        if digest(ROOT/config['model_path']/n)!=sha:raise ValueError('Qualified checkpoint changed')
    from scripts.run_i2_service_gate import audit_service
    for variant in config['variants']:
        actual=audit_service(l/f'service_{variant}',lm['config'])
        if not actual['passed'] or actual!=json.loads((l/f'service_{variant}/audit.json').read_text()):raise ValueError('HTTP evidence changed')
        policy=[r for r in json.loads((l/f'service_{variant}/events.json').read_text()) if r['event']=='i2_shape_policy']
        if len(policy)!=1 or policy[0]['name']!=config['policy'] or policy[0]['source_sha256']!={n:qm['source_sha256'][n] for n in ('serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py')}:raise ValueError('HTTP engine did not apply the qualified policy')
    return dict(qualification_audit_sha256=digest(q/'audit.json'),lifecycle_summary_sha256=digest(l/'summary.json'),
        reference_scope=spec['reference_scope'],compared_vectors=qa['compared_vectors'],performance_allowed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    if not args.attempt.replace('_','').isalnum():raise ValueError('Simple attempt required')
    lock=(ROOT/'runs/tmp/h2_runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'config/serving/i2_shape_performance.json';spec=json.loads(path.read_text())
    config=dict(json.loads((ROOT/spec['base_config']).read_text()),**{k:v for k,v in spec.items() if k!='base_config'})
    q=ROOT/spec['qualification_directory'];workload=next(s for s in json.loads((q/'workload.json').read_text())['suites'] if s['name']=='control512')['workload']
    gate=prerequisites(spec,config,workload)
    directory=ROOT/spec['results_dir']/args.attempt;directory.mkdir(parents=True,exist_ok=False)
    names=['config/serving/i2_shape_performance.json',spec['base_config'],'docs/i2_shape_performance_protocol.md',
        'scripts/run_i2_shape_performance.py','serving/i2_shape_worker.py','serving/i2_shape_runtime.py',
        'serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py','serving/i2_runtime.py','serving/i2_worker.py',
        'serving/i2_workloads.py','scripts/run_i2_single_gpu.py','scripts/prepare_i2_minisglang.py','scripts/prepare_i1_lifecycle.py']
    names+=['scripts/audit_i2_shape_performance.py','scripts/audit_i2_shape_qualification.py','scripts/audit_i2_shape_lifecycle.py',
        'scripts/run_i2_service_gate.py','scripts/launch_i2_shape_service.py','config/serving/i1_requirements.txt',
        'tests/test_i2_attention_reference.py','tests/test_i2_performance_gate.py']
    from scripts.prepare_i2_minisglang import prepare_i2
    config_sha=token_hash(config)
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (directory/'workload.json').write_text(json.dumps(workload,indent=2)+'\n')
    meta=dict(config=config,config_sha256=config_sha,workload_sha256=digest(directory/'workload.json'),series=prepare_i2(),
        source_sha256={n:digest(ROOT/n) for n in names},gates=gate,paid_resources_used=False)
    for n in names:
        dest=directory/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    (directory/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    (directory/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    from scripts.run_i2_single_gpu import launch
    journal=[];variants=list(config['variants'])
    for run in range(1,4):
        order=variants if run==1 else list(reversed(variants)) if run==2 else variants[3:]+variants[:3]
        for variant in order:
            launch([sys.executable,'-m','serving.i2_shape_worker','--directory',str(directory),'--variant',variant,'--run',str(run)],
                f'benchmark_{variant}_run{run}',config,args.attempt,directory,journal)


if __name__=='__main__':main()
