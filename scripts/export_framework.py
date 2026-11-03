"""Export audited native queue/KV/GPU samples and a standalone dashboard snapshot."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.delivery import read,sha,write
from serving.framework_metrics import HTML


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=a.directory.resolve()
    if (d/'export.json').exists():raise ValueError('Preserve the existing artifact export')
    audit=read(d/'audit.json')
    if not audit['passed'] or audit['scenario']!='healthy':raise ValueError('Independent healthy audit required')
    samples=[json.loads(line) for line in (d/'native_metrics_samples.jsonl').read_text().splitlines() if line]
    if not samples:raise ValueError('Actual native samples required')
    t0=samples[0]['monotonic_ns'];times=[(s['monotonic_ns']-t0)/1e9 for s in samples]
    if any(b<a for a,b in zip(times,times[1:])):raise ValueError('Non-monotonic sample clock')
    def values(key):return [s['native']['metrics'].get(key) if s['native']['metrics'].get(key) is not None else float('nan') for s in samples]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
    axes[0].plot(times,values('running'),label='Native running',color='#176b98')
    axes[0].plot(times,values('queued'),label='Native queued',color='#cb7627')
    axes[0].axhline(4,color='#176b98',linestyle=':',alpha=.5);axes[0].axhline(8,color='#cb7627',linestyle=':',alpha=.5)
    axes[0].set_ylabel('Requests');axes[0].legend(loc='upper right',ncol=2)
    axes[1].plot(times,values('KV_used'),label='KV actively used',color='#6a4b9b')
    axes[1].plot(times,values('KV_available'),label='KV available',color='#27835d')
    axes[1].set_ylabel('KV token slots');axes[1].legend(loc='upper right',ncol=2)
    axes[2].plot(times,[s['GPU_used_MiB'] for s in samples],label='GPU total, nvidia-smi',color='#294b6e')
    axes[2].set_ylabel('GPU memory / MiB');axes[2].set_xlabel('Seconds since first native metric sample (monotonic)')
    axes[2].legend(loc='upper right')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('SGLang 0.5.20 native BF16 / TP1: functional delivery observation\nRunning-service samples; post-shutdown GPU=0 is recorded separately')
    fig.savefig(d/'native_timeline.png',dpi=160);fig.savefig(d/'native_timeline.svg');plt.close(fig)
    frame=read(d/'dashboard_snapshot.json')
    encoded=json.dumps(frame,ensure_ascii=False).replace('</','<\\/')
    html=HTML.replace('<script>','<script>const frozenSnapshot='+encoded+';window.fetch=async()=>({json:async()=>frozenSnapshot});',1)
    html=html.replace('RouteScale · SGLang 原生指标','RouteScale · SGLang 原生指标（离线快照）')
    (d/'dashboard_snapshot.html').write_text(html)
    files=['native_timeline.png','native_timeline.svg','dashboard_snapshot.html']
    write(d/'export.json',dict(passed=True,audit_sha256=sha(d/'audit.json'),source_sha256=sha(Path(__file__)),
        samples=len(samples),elapsed_sample_s=times[-1],artifacts_sha256={name:sha(d/name) for name in files},performance_ranking_claimed=False))
    print('Exported',len(files),'artifacts from',len(samples),'actual samples')


if __name__=='__main__':main()
