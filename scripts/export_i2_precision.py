"""Export every accepted I2 cell, paired cost and comparison with static figures."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args(); directory=args.directory
    audit=json.loads((directory/'audit.json').read_text())
    if not audit['passed'] or not audit['local_single_gpu_complete']:
        raise ValueError('Independent final timing acceptance required')
    output=directory/'export.json'
    if output.exists():
        raise ValueError('Preserve report export')
    rows=[]
    for row in audit['statistics']:
        result={k:row[k] for k in ('mode','variant','case','phase','throughput_median','throughput_min','throughput_max','first_result_ms_median','process_cv')}
        result.update(prefill_queries=row['processes'][0]['prefill_queries'],
            prefill_linear_padded_rows_per_projection=row['processes'][0]['prefill_linear_padded_rows_per_projection'],
            external_HF_threshold_passed=row['mode']=='mixed')
        rows.append(result)
    for name,values in [('statistics.csv',rows),('precision_costs.csv',audit['precision_costs']),('comparisons.csv',audit['comparisons'])]:
        with (directory/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'svg.hashsalt':'i2-precision-v1','font.size':9})
    variants=['naive_full','radix_full','radix_chunk256','radix_chunk128','radix_graph124','radix_graph13','radix_overlap']
    labels=['Naive','Radix','Chunk256','Chunk128','Graph124','Graph13','Overlap']
    cells=[(case,phase) for case in ('b1_zero','b1_shared','b3_zero','b3_shared') for phase in ('cold','warm')]
    index={(r['mode'],r['variant'],r['case'],r['phase']):r for r in audit['statistics']}
    for metric,name,title,unit in [('throughput_median','throughput','Output throughput; median and range of 3 process means','output tokens/s'),
                                  ('first_result_ms_median','first_result_ms','Scheduler first-result; median and range of 3 process means','ms')]:
        fig,axes=plt.subplots(2,4,figsize=(16,7.6),constrained_layout=True)
        for ax,(case,phase) in zip(axes.flat,cells):
            for offset,mode,color,label in [(-.19,'bf16_control','#8c939d','BF16 control (fails external HF gate)'),(.19,'mixed','#168b83','FP16 main + FP32 residual/norm')]:
                values=[];low=[];high=[]
                for v in variants:
                    row=index[(mode,v,case,phase)]
                    points=[p['throughput' if metric=='throughput_median' else 'first_result_ms'] for p in row['processes']]
                    median=row[metric]
                    values.append(median);low.append(median-min(points));high.append(max(points)-median)
                ax.bar(np.arange(7)+offset,values,.36,color=color,label=label,yerr=[low,high],capsize=2,error_kw={'elinewidth':.7})
            ax.set_title(f'{case}, {phase}');ax.set_xticks(np.arange(7),labels,rotation=42,ha='right')
            ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        handles,legend=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='outside lower center',ncols=2)
        fig.suptitle(title+'\nFI/TP1; common prefill no-split + linear tile512; fixed 64-step replay; not online SLO')
        for suffix in ('svg','png'):
            fig.savefig(directory/f'{name}.{suffix}',dpi=160,metadata={'Date':None} if suffix=='svg' else None)
        plt.close(fig)
    cost={(r['variant'],r['case'],r['phase']):r for r in audit['precision_costs']}
    fig,axes=plt.subplots(2,4,figsize=(16,7.2),constrained_layout=True)
    for ax,(case,phase) in zip(axes.flat,cells):
        values=[cost[(v,case,phase)] for v in variants]
        median=[r['mixed_over_BF16_throughput_median'] for r in values]
        ax.errorbar(np.arange(7),median,yerr=[[m-r['ratio_min'] for m,r in zip(median,values)],
            [r['ratio_max']-m for m,r in zip(median,values)]],fmt='o',color='#168b83',capsize=3)
        ax.axhline(1,color='#666',linestyle='--',linewidth=.8)
        ax.set_xticks(np.arange(7),labels,rotation=42,ha='right');ax.set_title(f'{case}, {phase}')
        ax.set_ylabel('mixed / BF16 throughput');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Precision-policy cost: median and range of 3 paired process ratios\nCommon shapes; BF16 control fails external HF threshold; higher ratio means greater retained throughput')
    for suffix in ('svg','png'):
        fig.savefig(directory/f'precision_costs.{suffix}',dpi=160,metadata={'Date':None} if suffix=='svg' else None)
    plt.close(fig)
    temperatures=[];clocks=[];powers=[];pstates=set()
    for path in directory.glob('benchmark_*.json'):
        for row in json.loads(path.read_text())['replay']['cases']:
            fields=[s.strip() for s in row['telemetry_after'].split(',')]
            temperatures.append(float(fields[2]));clocks.append(float(fields[3].split()[0]))
            powers.append(float(fields[5].split()[0]));pstates.add(fields[6])
    record=dict(audit_sha256=digest(directory/'audit.json'),source_sha256=digest(Path(__file__)),
        packages={n:importlib.metadata.version(n) for n in ('matplotlib','numpy')},statistics_cells=len(rows),
        precision_cost_cells=len(audit['precision_costs']),system_comparisons=len(audit['comparisons']),
        process_cv_min=min(r['process_cv'] for r in rows),process_cv_max=max(r['process_cv'] for r in rows),
        telemetry_after_wave=dict(samples=len(temperatures),temperature_c=[min(temperatures),max(temperatures)],
            sm_clock_mhz=[min(clocks),max(clocks)],power_w=[min(powers),max(powers)],pstates=sorted(pstates),
            boundary='after wave/resource audit; not a kernel-clock measurement'),
        artifact_sha256={p.name:digest(p) for p in directory.iterdir() if p.suffix in ('.svg','.png','.csv')})
    output.write_text(json.dumps(record,indent=2)+'\n')
    target=directory/'export_source_snapshot/scripts'
    target.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),target/Path(__file__).name)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
