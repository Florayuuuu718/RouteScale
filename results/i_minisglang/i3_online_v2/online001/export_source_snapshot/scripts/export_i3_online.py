"""Complete per-request/run/cell CSV exports and finite-trace online figures."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i3_workloads import read,write
from serving.i2_workloads import digest


def flatten(value,prefix=''):
    output={}
    for key,item in value.items():
        name=prefix+key
        if isinstance(item,dict):output.update(flatten(item,name+'.'))
        else:output[name]=json.dumps(item) if isinstance(item,list) else item
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    directory=args.directory;a=read(directory/'audit.json');target=directory/'export.json'
    if not a['passed']:raise ValueError('Positive independent online audit required')
    if target.exists():raise ValueError('Preserve export')
    for name,values in [('requests.csv',a['requests']),('runs.csv',a['runs']),
                        ('statistics.csv',a['statistics']),('comparisons.csv',a['comparisons'])]:
        rows=[flatten(v) for v in values]
        with (directory/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'svg.hashsalt':'i3-online-v1','font.size':10})
    levels=['low','near','overload'];variants=['radix','graph124']
    colors=['#8c939d','#168b83'];labels=['Radix (v2 precision)','Graph[1,2,4] (v2 precision)']
    cells={(r['variant'],r['rate']):r for r in a['statistics']}
    ticks=[f"{name}\n{a['rates_rps'][name]:.3f} req/s" for name in levels]
    def bars(ax,key,quantile=None):
        for offset,variant,color,label in zip([-.2,.2],variants,colors,labels):
            rows=[cells[variant,rate][key] if quantile is None else cells[variant,rate][key][quantile] for rate in levels]
            vals=[r['median'] if r['median'] is not None else np.nan for r in rows]
            low=[v-r['min'] if r['min'] is not None else np.nan for r,v in zip(rows,vals)]
            high=[r['max']-v if r['max'] is not None else np.nan for r,v in zip(rows,vals)]
            ax.bar(np.arange(3)+offset,vals,.36,color=color,label=label,yerr=[low,high],capsize=3)
        ax.set_xticks(np.arange(3),ticks);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    def save(fig,name):
        for suffix in ('svg','png'):
            fig.savefig(directory/f'{name}.{suffix}',dpi=160,metadata={'Date':None} if suffix=='svg' else None)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
    for ax,key,title in zip(axes,['completed_request_throughput','goodput_rps','offered_SLO_success_fraction'],
        ['Completed request throughput (req/s)','All-SLO goodput (req/s)','All-SLO fraction of every offered request']):
        bars(ax,key);ax.set_title(title)
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncols=2)
    fig.suptitle('I3 finite open-arrival trace: 48 offers/process; median and range of 3 independent services\nGoodput: first text <=1s, E2E <=8s, amortized TPOT <=100ms; throughput includes drain')
    save(fig,'throughput_goodput')
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    specs=[('client_text_ttft_ms','Client first-text p95 (ms); successes only',1000),
        ('e2e_ms','Client E2E p95 (ms); successes only',8000),
        ('amortized_tpot_ms','Amortized TPOT p95 (ms); successes only',100),
        ('scheduler_queue_ms','Scheduler queue p95 (ms); scheduled requests',None),
        ('backend_emission_itl_ms','Backend host emission ITL p95 (ms)',None),
        ('sse_text_chunk_intervals_ms','SSE text-chunk interval p95 (ms)',None)]
    for ax,(key,title,goal) in zip(axes.flat,specs):
        bars(ax,key,'p95');ax.set_title(title)
        if goal is not None:ax.axhline(goal,color='#b55',linestyle='--',linewidth=.8)
    fig.legend(handles,legend,loc='outside lower center',ncols=2)
    fig.suptitle('I3 conditional latency: median/range of 3 process p95 values\nClient text chunks are not token timestamps; failed/rejected requests are retained in the offered ledger')
    save(fig,'latency')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    states=['finished','rejected','cancelled','failed'];state_colors=['#168b83','#d29448','#9676aa','#c95c5c']
    for ax,variant,label in zip(axes,variants,labels):
        bottoms=np.zeros(3)
        for state,color in zip(states,state_colors):
            values=[sum(r['terminal_counts'][state]/48 for r in a['runs'] if r['variant']==variant and r['rate']==rate)/3 for rate in levels]
            ax.bar(np.arange(3),values,.6,bottom=bottoms,color=color,label=state);bottoms+=values
        slo=[cells[variant,rate]['offered_SLO_success_fraction']['median'] for rate in levels]
        ax.plot(np.arange(3),slo,'ko',label='median all-SLO fraction')
        ax.set_xticks(np.arange(3),ticks);ax.set_ylim(0,1.08);ax.set_ylabel('fraction of all 48 offered requests');ax.set_title(label)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=5)
    fig.suptitle('I3 complete offered-request accounting\nStacked fractions are means over 3 independent services; every offer has exactly one terminal')
    save(fig,'accounting')
    record=dict(audit_sha256=digest(directory/'audit.json'),source_sha256=digest(Path(__file__)),
        packages={n:importlib.metadata.version(n) for n in ('matplotlib','numpy')},
        offered_records=len(a['requests']),formal_processes=len(a['runs']),statistics_cells=len(a['statistics']),
        artifact_sha256={p.name:digest(p) for p in directory.iterdir() if p.suffix in ('.csv','.svg','.png')})
    write(target,record);dest=directory/'export_source_snapshot/scripts';dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),dest/Path(__file__).name)
    print('Exported',record['offered_records'],'requests /',record['formal_processes'],'runs /',record['statistics_cells'],'cells')


if __name__=='__main__':main()
