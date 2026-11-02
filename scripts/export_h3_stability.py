"""Export accepted H3 replication traces, statistics and uncertainty figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.backends.nanovllm import digest, write_result
from serving.h3_stability import config, read


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    directory = ROOT / config()['results_dir']; audit = read(directory / 'audit.json')
    if not audit['investigation_complete']: raise ValueError('Positive independent audit required')
    if (directory / 'export.json').exists(): raise ValueError('Preserve exports')
    rows = dict(processes=audit['process_scope_rows'], statistics=audit['statistics'], comparisons=audit['comparisons'])
    wave_rows = []; sensor_rows = []
    for path in sorted((directory / 'raw').glob('*.json')):
        r = read(path)
        for scope, m in r['measurements'].items():
            for phase,key in [('warmup','warmup_traces'),('measurement','traces')]:
                for index, t in enumerate(m[key]):
                    wave_rows.append(dict(label=r['job']['label'], scope=scope, phase=phase, wave=index+1, **t))
        sensor_rows.extend(dict(job_label=r['job']['label'], **s) for s in r['sensors'])
    rows.update(waves=wave_rows, sensors=sensor_rows); artifacts = []
    for name, records in rows.items():
        p = directory / f'{name}.csv'
        with p.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader()
            writer.writerows({k: json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records)
        artifacts.append(p)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants = ['native_kv','paged_eager','paged_graph']; colors = ['#587d9c','#e19b45','#3e9b7a']
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,case in enumerate(config()['case_ids']):
        for j,scope in enumerate(['model_execution','replay_e2e']):
            ax=axes[i,j]
            for index,(variant,color) in enumerate(zip(variants,colors)):
                row=next(r for r in audit['statistics'] if r['case_id']==case and r['variant']==variant and r['measurement_scope']==scope)
                values=row['throughput_values']; x=[index+(k-2.5)*.025 for k in range(6)]
                ax.scatter(x,values,color=color,s=28)
                ax.hlines(row['output_tokens_per_second'],index-.22,index+.22,color=color,lw=2)
            ax.set_xticks(range(3),['Native KV','Paged eager','Paged Graph']); ax.set_ylabel('Output tokens/s')
            ax.set_title(case+' / '+scope); ax.grid(axis='y',alpha=.2)
    fig.suptitle('H3 v5 replication: six fresh processes per cell\nDots: process medians; lines: median of six; fixed-token replay')
    for ext in ['png','svg']:
        p=directory/f'throughput.{ext}'; fig.savefig(p,dpi=160);artifacts.append(p)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,case in zip(axes,config()['case_ids']):
        for scope,marker in [('model_execution','o'),('replay_e2e','s')]:
            for baseline,color in [('paged_eager','#3e9b7a'),('native_kv','#587d9c')]:
                row=next(r for r in audit['comparisons'] if r['case_id']==case and r['measurement_scope']==scope and r['baseline']==baseline and r['candidate']=='paged_graph')
                ax.plot(range(1,7),row['paired_ratios'],marker=marker,color=color,ls='-' if scope=='replay_e2e' else '--',label=f'Graph / {baseline}, {scope}')
        ax.axhline(1,color='gray',lw=.8);ax.set_title(case);ax.set_xlabel('Balanced independent round');ax.set_ylabel('Paired throughput ratio');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle('Paired ratios across all six rounds; no outlier exclusion')
    for ext in ['png','svg']:
        p=directory/f'paired_ratios.{ext}';fig.savefig(p,dpi=160);artifacts.append(p)
    plt.close(fig)
    write_result(dict(audit_sha256=digest(directory/'audit.json'), source_sha256=digest(Path(__file__)),
        matplotlib=matplotlib.__version__, CSV_rows={k:len(v) for k,v in rows.items()},
        artifact_sha256={p.name:digest(p) for p in artifacts}),directory/'export.json')
    print('H3 export complete:', {k:len(v) for k,v in rows.items()})


if __name__=='__main__': main()
