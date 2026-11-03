"""Create an offline SGLang container context from verified locked dependencies.

Weights are never copied. Archives and the resulting image stay outside Git.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.delivery import sha,write
from serving.sglang_framework import doctor,load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    out=a.directory.resolve()
    if out.exists():raise ValueError('Fresh offline build context required')
    if subprocess.check_output(['git','status','--porcelain'],text=True,cwd=ROOT).strip():raise ValueError('Commit the source before creating its container context')
    c=load();checked=doctor(c);out.mkdir(parents=True)
    files=[n for n in ['scripts/framework_service.py','scripts/check_framework.py','scripts/audit_framework.py',
        'config/delivery/model.json','docs/sglang_delivery_protocol.md',
        'results/i_minisglang/i3_online_v2/workload002/workload.json','results/i_minisglang/i3_online_v2/online001/schedule.json']]
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'serving').glob('*.py')]
    files += [str(f.relative_to(ROOT)) for f in (ROOT/'deploy/local/sglang').iterdir() if f.is_file()]
    for name in files:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    site=ROOT/c['environment']/'lib/python3.11/site-packages'
    subprocess.run(['tar','--exclude=__pycache__','-C',str(site),'-cf',str(out/'sglang-environment.tar'),'.'],check=True)
    toolkit=Path('/usr/local/cuda-13.0')
    subprocess.run(['tar','--dereference','-C',str(toolkit),'-cf',str(out/'cuda-toolchain.tar'),
        'bin','targets/x86_64-linux/include','targets/x86_64-linux/lib','nvvm'],check=True)
    # Retain the declared local header compatibility fix in the portable toolkit.
    header='targets/x86_64-linux/include/crt/math_functions.h'
    subprocess.run(['tar','-C',str(ROOT/'runs/toolchains/cuda-13.0-i1'),'-rf',str(out/'cuda-toolchain.tar'),header],check=True)
    write(out/'framework_bundle.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        clean_source_checkout=True,source_sha256={n:sha(ROOT/n) for n in files},doctor=checked,
        requirements_sha256=sha(ROOT/'deploy/local/sglang/requirements.lock'),toolchain_header_sha256=sha(ROOT/'runs/toolchains/cuda-13.0-i1'/header),
        archives_sha256={n:sha(out/n) for n in ['sglang-environment.tar','cuda-toolchain.tar']},weights_embedded=False))
    print(json.dumps(dict(directory=str(out),weights_embedded=False,passed=True)))


if __name__=='__main__':main()
