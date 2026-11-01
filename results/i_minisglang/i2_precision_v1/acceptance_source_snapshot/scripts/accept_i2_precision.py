"""Bind final I2 local gates, export hashes, CPU checks and clean GPU shutdown."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def main():
    import psutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance',type=Path,required=True)
    args=parser.parse_args(); directory=args.performance.resolve()
    base=directory.parent;output=base/'acceptance.json'
    if output.exists():raise ValueError('Preserve acceptance')
    read=lambda p:json.loads(p.read_text())
    report=read(directory/'audit.json');meta=read(directory/'provenance.json');config=meta['config']
    q=ROOT/meta['artifact_bindings']['qualification_directory'];l=ROOT/meta['artifact_bindings']['lifecycle_directory']
    qa,la=read(q/'audit.json'),read(l/'audit.json');export=read(directory/'export.json')
    cpu=read(base/'cpu_tests.json')
    shutdown_cpu=read(base/'shutdown_cpu_tests.json');fault=read(base/'shutdown_fault_probe.json')
    followup_cpu=read(base/'followup_cpu_tests.json')
    checks=dict(performance_acceptance=report['passed'] and report['local_single_gpu_complete'] and report['formal_processes']==42,
        numerical_acceptance=qa['qualification_passed'] and qa['compared_vectors']==25088 and qa['mixed_external_HF_vectors']==12544,
        HTTP_acceptance=la['HTTP_lifecycle_passed'] and la['requests']==168 and la['scenario_checks']==560,
        CPU_acceptance=cpu['passed'] and cpu['tests']==9 and digest(base/'cpu_tests.txt')==cpu['log_sha256'],
        CPU_sources_unchanged=all(digest(ROOT/n)==sha==digest(base/'cpu_source_snapshot'/n) for n,sha in cpu['source_sha256'].items()),
        shutdown_CPU_tests=shutdown_cpu['passed'] and shutdown_cpu['tests']==2 and all(digest(ROOT/n)==sha for n,sha in shutdown_cpu['source_sha256'].items()),
        retry_gate_CPU_tests=followup_cpu['passed'] and followup_cpu['tests']==9
            and digest(base/'followup_cpu_tests.txt')==followup_cpu['log_sha256']
            and all(digest(ROOT/n)==sha==digest(base/'followup_cpu_source_snapshot'/n) for n,sha in followup_cpu['source_sha256'].items()),
        real_fault_budget=fault['passed'] and fault['report']['seconds']<10,
        shutdown_audit=read(l/'shutdown_audit.json')['passed'],
        exported_all_cells=export['statistics_cells']==112 and export['precision_cost_cells']==56 and export['system_comparisons']==96,
        export_binding=export['audit_sha256']==digest(directory/'audit.json'),
        export_hashes=all(digest(directory/n)==sha for n,sha in export['artifact_sha256'].items()),
        historical_failure_preserved=read(ROOT/'results/i_minisglang/i2_single_gpu_v1/attempt001/numerical_gate.json')['passed'] is False,
        BF16_failure_not_relabelled=report['original_BF16_external_HF_passed'] is False)
    journals={}
    for name,path in [('qualification',q),('HTTP',l),('performance',directory)]:
        journal=read(path/'processes.json')
        checks[name+'_no_process_residue']=all(not psutil.pid_exists(r['pid']) for r in journal)
        checks[name+'_source_snapshots']=all(digest(path/'source_snapshot'/n)==sha for n,sha in read(path/'provenance.json')['source_sha256'].items())
        auditor=path/'audit_source_snapshot/scripts'/('audit_i2_precision_final.py' if name=='performance' else 'audit_i2_precision.py')
        checks[name+'_auditor_snapshot']=digest(auditor)==read(path/'audit.json')['auditor_sha256']
        journals[name]=dict(processes=len(journal),sha256=digest(path/'processes.json'))
    from scripts.prepare_i2_minisglang import validate
    series=validate(ROOT/config['checkout'])
    checks['unchanged_native_tree']=series==meta['series']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    checks['GPU_idle']=gpu=='0 MiB, 0 %'
    passed=all(checks.values())
    result=dict(stage='I2',scope=report['scope'],local_single_gpu_complete=passed,checks=checks,
        numerical_vectors=dict(mixed_external_HF=12544,BF16_internal_control=12544),
        HTTP_requests=168,HTTP_checks=560,formal_processes=42,CPU_tests=20,
        original_BF16_external_HF_passed=False,TP2_4_complete=False,paid_resources_used=False,
        GPU_end=gpu,process_journals=journals,
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            (q/'audit.json',l/'audit.json',directory/'audit.json',directory/'export.json',base/'cpu_tests.json')},
        source_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result,indent=2)+'\n')
    snapshot=base/'acceptance_source_snapshot/scripts';snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print(json.dumps(result,indent=2))
    if not passed:raise ValueError('Local I2 acceptance incomplete')


if __name__=='__main__':main()
