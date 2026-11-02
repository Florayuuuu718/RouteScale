"""Replay all nine patches into a standalone, clean delivery checkout."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import load,read
from scripts.prepare_i2_minisglang import validate


def prepare(seed):
    target=ROOT/load()['checkout']
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(seed),str(target)],check=True)
        base=read(ROOT/'patches/minisgl-i1/series.json')['base_commit']
        subprocess.run(['git','-C',str(target),'checkout','--detach',base],check=True)
        patches=[]
        for stage in ('minisgl-i1','minisgl-i2'):
            series=read(ROOT/f'patches/{stage}/series.json')
            patches += [str(ROOT/'patches'/stage/name) for name in series['patch_sha256']]
        subprocess.run(['git','-C',str(target),'-c','user.name=RouteScale','-c','user.email=routescale@localhost',
            '-c','commit.gpgSign=false','am',*patches],check=True)
    print(validate(target))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',default='https://github.com/sgl-project/mini-sglang.git',help='Pinned upstream URL or an offline local Git repository')
    prepare(p.parse_args().seed)
