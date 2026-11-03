"""SGLang formal framework quickstart, readiness and local dashboard."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.sglang_framework import CONFIG,doctor,load,serve
from serving.delivery import read,request_stop


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['doctor','run','status','stop'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--run-dir',type=Path)
    a=p.parse_args();c=load(a.config)
    if a.action=='doctor':print(json.dumps(doctor(c),indent=2));return 0
    if a.run_dir is None:p.error('--run-dir required')
    if a.action=='run':return serve(c,a.run_dir)
    if a.action=='stop':
        # The shared identity guard expects the configured shutdown budget.
        print(json.dumps(request_stop(a.run_dir),indent=2));return 0
    print(json.dumps(read(a.run_dir/'status.json'),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
