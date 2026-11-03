"""Create an offline SGLang container context from verified locked dependencies.

Weights are never copied. Archives and the resulting image stay outside Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.delivery import sha,write
from serving.sglang_framework import doctor,load


def archive_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    out=a.directory.resolve()
    if out.exists():raise ValueError('Fresh offline build context required')
    if subprocess.check_output(['git','status','--porcelain'],text=True,cwd=ROOT).strip():raise ValueError('Commit the source before creating its container context')
    c=load();checked=doctor(c);out.mkdir(parents=True)
    files=[n for n in ['scripts/framework_service.py','scripts/launch_sglang_framework.py','scripts/check_framework.py','scripts/audit_framework.py',
        'config/delivery/model.json','docs/sglang_delivery_protocol.md',
        'results/i_minisglang/i3_online_v2/workload002/workload.json','results/i_minisglang/i3_online_v2/online001/schedule.json']]
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'serving').glob('*.py')]
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'deploy/local/sglang').iterdir() if f.is_file()]
    for name in files:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    console=out/'console-scripts';console.mkdir();scripts={}
    for source in (ROOT/c['environment']/'bin').iterdir():
        if source.name.startswith('python') or not source.is_file() or not os.access(source,os.X_OK):continue
        value=source.read_bytes()
        if value.startswith(b'#!') and b'python' in value.split(b'\n',1)[0]:value=b'#!/usr/local/bin/python3\n'+value.split(b'\n',1)[1]
        target=console/source.name;target.write_bytes(value);target.chmod(0o755);scripts[source.name]=sha(target)
    if 'ninja' not in scripts:raise ValueError('Native compiler requires the pinned Ninja executable')
    header='targets/x86_64-linux/include/crt/math_functions.h'
    # Small tar writes through WSL/NTFS are expensive. Stage/compress on Linux,
    # then copy each archive in large blocks to the explicit build directory.
    staging=ROOT/'runs/tmp';staging.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sglang-bundle-',dir=staging) as temp:
        temp=Path(temp)
        for name,source,entries,options in [
            ('sglang-environment',ROOT/c['environment']/'lib/python3.11/site-packages',['.'],['--exclude=__pycache__']),
            ('cuda-toolchain',Path('/usr/local/cuda-13.0'),['bin','targets/x86_64-linux/include','targets/x86_64-linux/lib','nvvm','EULA.txt','README','version.json'],['--dereference'])]:
            raw=temp/(name+'.tar');packed=temp/(name+'.tar.gz')
            subprocess.run(['tar','--blocking-factor=2048',*options,'-C',str(source),'-cf',str(raw),*entries],check=True)
            if name=='cuda-toolchain':
                subprocess.run(['tar','-C',str(ROOT/'runs/toolchains/cuda-13.0-i1'),'-rf',str(raw),header],check=True)
            with packed.open('wb') as stream:subprocess.run(['gzip','-1','-n','-c',str(raw)],stdout=stream,check=True)
            with packed.open('rb') as src,(out/packed.name).open('xb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
            print('Prepared',packed.name,flush=True)
            raw.unlink();packed.unlink()
    write(out/'framework_bundle.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        clean_source_checkout=True,source_sha256={n:sha(ROOT/n) for n in files},doctor=checked,
        requirements_sha256=sha(ROOT/'deploy/local/sglang/requirements.lock'),toolchain_header_sha256=sha(ROOT/'runs/toolchains/cuda-13.0-i1'/header),
        archives_sha256={n:archive_sha(out/n) for n in ['sglang-environment.tar.gz','cuda-toolchain.tar.gz']},console_scripts_sha256=scripts,weights_embedded=False))
    print(json.dumps(dict(directory=str(out),weights_embedded=False,passed=True)))


if __name__=='__main__':main()
