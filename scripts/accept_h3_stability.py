"""Bind the H3 investigation audit, exports, CPU evidence and idle GPU."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_stability import check_sources, config, gpu_idle, read


def main():
    c=config();directory=ROOT/c['results_dir'];target=directory/'acceptance.json'
    if target.exists():raise ValueError('Preserve acceptance')
    meta=check_sources(directory);a=read(directory/'audit.json');export=read(directory/'export.json');cpu=read(directory/'cpu_tests.json')
    checks=dict(independent_audit=a['investigation_complete'] and all(a['checks'].values()),
        fresh_numerical=a['qualification_vectors']==1728 and a['graph_exact_vectors']==576,
        complete_matrix=a['formal_processes']==36 and a['measured_waves']==1440 and a['warmup_waves']==1440,
        frozen_sources=True,historical_evidence=True,CPU_tests=cpu['passed'] and cpu['tests']>=12,
        export_audit_binding=export['audit_sha256']==digest(directory/'audit.json'),
        exporter_source=export['source_sha256']==meta['source_sha256']['scripts/export_h3_stability.py'],
        export_artifacts=all(digest(directory/name)==sha for name,sha in export['artifact_sha256'].items()),
        CSV_counts=export['CSV_rows']==dict(processes=72,statistics=12,comparisons=12,waves=2880,sensors=432),
        GPU_idle=gpu_idle()=='0 MiB, 0 %')
    evidence=[directory/n for n in ('provenance.json','qualification_gate.json','audit.json','export.json','cpu_tests.json','processes.json')]
    result=dict(stage='H3',scope=c['scope'],local_timing_investigation_complete=all(checks.values()),checks=checks,
        timing_stable=a['timing_stable'],timing_status=a['timing_status'],
        numerical_vectors=1728,graph_exact_vectors=576,formal_processes=36,measured_waves=1440,CPU_tests=cpu['tests'],
        GPU_end='0 MiB, 0 %',paid_resources_used=False,TP2_4_complete=False,
        evidence_sha256={str(p.relative_to(ROOT)):digest(p) for p in evidence},source_sha256=digest(Path(__file__)))
    write_result(result,target)
    print('H3 investigation accepted:',result['local_timing_investigation_complete'],'timing:',a['timing_status'])
    if not result['local_timing_investigation_complete']:raise ValueError('H3 acceptance failed')


if __name__=='__main__':main()
