"""Render process ranges; requires the isolated runs/envs/i2-report environment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.directory/'audit.json').read_text())
    if not report['passed']:raise ValueError('Only accepted process evidence can be plotted')
    rows=report['statistics']
    names=['naive_full','radix_full','radix_chunk256','radix_chunk128','radix_graph124','radix_graph13','radix_overlap']
    labels=['Naive','Radix','Chunk 256','Chunk 128','Graph 1/2/4','Graph 1/3','Overlap']
    indexed={(r['variant'],r['case'],r['phase']):r for r in rows}
    for metric,title,unit in [('throughput','Fixed-token scheduler replay throughput','output token/s'),
                              ('first_result_ms','Scheduler first result (includes Prefill)','ms; excludes HTTP')]:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
        for row,batch in enumerate((1,3)):
            for col,phase in enumerate(('cold','warm')):
                ax=axes[row,col]
                for offset,prefix,color in [(-.18,'zero','#577590'),(.18,'shared','#43aa8b')]:
                    samples=[[p[metric] for p in indexed[(name,f'b{batch}_{prefix}',phase)]['processes']] for name in names]
                    med=np.array([np.median(v) for v in samples])
                    low=med-np.array([min(v) for v in samples]);high=np.array([max(v) for v in samples])-med
                    x=np.arange(len(names))+offset
                    ax.bar(x,med,.34,color=color,label=f'{prefix} prefix',alpha=.75,yerr=np.stack([low,high]),capsize=3)
                    for i,values in enumerate(samples):
                        ax.scatter(x[i]+np.linspace(-.045,.045,len(values)),values,s=12,color='#263238',zorder=4)
                ax.set_title(f'batch={batch}; prefix cache {phase}')
                ax.set_xticks(np.arange(len(names)),labels,rotation=30,ha='right')
                ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                if row==0 and col==0:ax.legend(frameon=False)
        fig.suptitle(title+'\n3 independent processes; median and min/max of process means',fontsize=14)
        for suffix in ('svg','png'):fig.savefig(args.directory/f'{metric}.{suffix}',dpi=180)
        plt.close(fig)
    evidence=dict(plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256((args.directory/'audit.json').read_bytes()).hexdigest(),
        matplotlib=matplotlib.__version__,numpy=np.__version__,description='Dots are independent process means; whiskers are observed min/max, not confidence intervals')
    (args.directory/'figures.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
