"""Resume large pinned wheels by HTTP ranges; verify official PyPI SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
LARGE=('sglang-kernel','nvidia-cublas','nvidia-cudnn-cu13','nvidia-cufft','nvidia-cusolver',
    'nvidia-cusparse','nvidia-cusparselt-cu13','nvidia-nccl-cu13')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def select(name,version):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as r:meta=json.load(r)
    wheels=[r for r in meta['urls'] if r['filename'].endswith('x86_64.whl') and 'manylinux' in r['filename']
        and ('-py3-none-' in r['filename'] or '-cp311-cp311-' in r['filename'] or '-cp310-abi3-' in r['filename'])]
    if len(wheels)!=1:raise ValueError(f'Ambiguous supported wheel: {name}: {[r["filename"] for r in wheels]}')
    w=wheels[0]
    if not w['url'].startswith('https://files.pythonhosted.org/'):raise ValueError('Official file host required')
    return dict(package=name,version=version,filename=w['filename'],url=w['url'],bytes=w['size'],sha256=w['digests']['sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=16:p.error('workers must be 1..16')
    out=a.directory.resolve();out.mkdir(parents=True,exist_ok=True)
    pins=dict(line.split('==',1) for line in (ROOT/'deploy/local/sglang/requirements.lock').read_text().splitlines() if line and not line.startswith('#'))
    manifest=out/'pypi_wheels.json'
    if manifest.exists():
        specs=json.loads(manifest.read_text())
        if {s['package']:s['version'] for s in specs}!={n:pins[n] for n in LARGE}:raise ValueError('Preserve previous wheel set; select a fresh directory')
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:specs=list(pool.map(lambda n:select(n,pins[n]),LARGE))
        manifest.write_text(json.dumps(specs,indent=2)+'\n')
    pending=[];chunk=4*1024*1024
    for s in specs:
        target=out/s['filename']
        if target.exists() and target.stat().st_size==s['bytes'] and sha(target)==s['sha256']:continue
        for start in range(0,s['bytes'],chunk):pending.append((s,start,min(s['bytes']-1,start+chunk-1)))
    def fetch(job):
        s,start,end=job;folder=out/'chunks'/s['filename'];folder.mkdir(parents=True,exist_ok=True)
        dest=folder/str(start)
        if dest.exists() and dest.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                req=urllib.request.Request(s['url'],headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=30) as r:
                    expected=f'bytes {start}-{end}/{s["bytes"]}'
                    if r.status!=206 or r.headers.get('Content-Range')!=expected:raise ValueError('Unexpected byte-range response')
                    value=r.read(end-start+2)
                if len(value)!=end-start+1:raise ValueError('Incomplete chunk')
                temp=dest.with_suffix('.tmp');temp.write_bytes(value);temp.replace(dest);return
            except Exception:
                if attempt==4:raise
                time.sleep(min(attempt+1,4))
    print('Official SHA-bound wheel chunks:',len(pending),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(fetch,job) for job in pending]),1):
            future.result()
            if i%20==0 or i==len(pending):print('Verified-length chunks',i,'/',len(pending),flush=True)
    results=[]
    for s in specs:
        target=out/s['filename']
        if not target.exists():
            temp=target.with_suffix('.assembling')
            with temp.open('wb') as f:
                for start in range(0,s['bytes'],chunk):f.write((out/'chunks'/s['filename']/str(start)).read_bytes())
            temp.replace(target)
        if sha(target)!=s['sha256']:raise ValueError('Full official SHA256 mismatch: '+s['filename'])
        results.append(dict(filename=s['filename'],sha256=s['sha256'],passed=True))
    (out/'verified.json').write_text(json.dumps(dict(passed=True,wheels=results),indent=2)+'\n')
    print('All eight official wheel SHA256s passed',flush=True)


if __name__=='__main__':main()
