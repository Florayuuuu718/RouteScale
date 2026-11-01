"""Bind final I3 independent gates, CPU checks, exports and clean shutdown."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def main():
    import psutil
    from scripts.prepare_i2_minisglang import validate
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    directory=args.directory.resolve();base=directory.parent;target=base/'acceptance.json'
    if target.exists():raise ValueError('Preserve final acceptance')
    a=read(directory/'audit.json');meta=read(directory/'provenance.json');binding=meta['artifact_bindings']
    q=ROOT/binding['qualification_directory'];pilot=ROOT/binding['pilot_directory']
    qa=read(q/'audit.json');pa=read(pilot/'audit.json');export=read(directory/'export.json');cpu=read(base/'cpu_tests.json')
    checks=dict(final_audit=a['passed'] and a['local_single_gpu_complete'] and all(a['checks'].values()),
        numerical=qa['passed'] and qa['vectors']==3840,diagnostic_pilot=pa['passed'],
        complete_matrix=a['formal_processes']==18 and a['offered_requests']==864 and len(a['statistics'])==6,
        CPU_tests=cpu['passed'] and cpu['tests']==26 and digest(base/'cpu_tests.txt')==cpu['log_sha256'],
        CPU_sources=all(digest(ROOT/n)==sha==digest(base/'cpu_source_snapshot'/n) for n,sha in cpu['source_sha256'].items()),
        exports=export['offered_records']==864 and export['formal_processes']==18 and export['statistics_cells']==6
            and export['audit_sha256']==digest(directory/'audit.json'),
        export_artifacts=all(digest(directory/n)==sha for n,sha in export['artifact_sha256'].items()),
        original_I2=read(ROOT/meta['config']['I2_acceptance'])['local_single_gpu_complete']
            and digest(ROOT/meta['config']['I2_acceptance'])==meta['I2_acceptance_sha256'],
        unchanged_native_tree=validate(ROOT/meta['config']['checkout'])==meta['series'])
    journals={};shutdowns=[]
    for name,path,n,auditor in [('pilot',pilot,2,'audit_i3_pilot.py'),('qualification',q,4,'audit_i3_qualification.py'),('formal',directory,18,'audit_i3_online.py')]:
        rows=read(path/'processes.json');journals[name]=dict(processes=len(rows),sha256=digest(path/'processes.json'))
        checks[name+'_processes']=len(rows)==n and all(not psutil.pid_exists(r['pid']) and r['status']=='pass' and r['returncode']==0 and r['gpu_after']=='0 MiB' for r in rows)
        pm=read(path/'provenance.json')
        checks[name+'_sources']=all(digest(ROOT/source)==sha==digest(path/'source_snapshot'/source) for source,sha in pm['source_sha256'].items())
        checks[name+'_auditor']=digest(path/'audit_source_snapshot/scripts'/auditor)==read(path/'audit.json')['auditor_sha256']==digest(ROOT/'scripts'/auditor)
        service_runs=[p for p in path.glob('*/run.json')]
        checks[name+'_service_shutdown']=all(not psutil.pid_exists(read(p)['server_pid']) and read(p)['returncode'] in (0,-15)
            and read(p).get('shutdown_request')=='SIGTERM'
            and not read(p)['forced_cleanup'] and not read(p)['surviving_pids'] and read(p)['shutdown_s']<=10 for p in service_runs)
        shutdowns += [read(p)['shutdown_s'] for p in service_runs]
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    checks['GPU_idle']=gpu=='0 MiB, 0 %'
    passed=all(checks.values())
    evidence=[q/'audit.json',pilot/'audit.json',directory/'audit.json',directory/'export.json',base/'cpu_tests.json']
    result=dict(stage='I3',scope=a['scope'],local_single_gpu_complete=passed,checks=checks,
        qualification_vectors=3840,formal_processes=18,offered_requests=864,CPU_tests=26,statistics_cells=6,
        finite_trace=True,steady_state_claimed=False,TP2_4_complete=False,paid_resources_used=False,
        GPU_end=gpu,max_service_shutdown_s=max(shutdowns),process_journals=journals,
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},source_sha256=digest(Path(__file__)))
    write(target,result);dest=base/'acceptance_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),dest/Path(__file__).name)
    print('I3 final acceptance:',passed,'failed:',[k for k,v in checks.items() if not v],flush=True)
    if not passed:raise ValueError('I3 acceptance incomplete')


if __name__=='__main__':main()
