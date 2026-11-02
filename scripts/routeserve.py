"""Local serving CLI: doctor, foreground run, status and identity-checked stop."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.delivery import DEFAULT,doctor,load,read,request_stop,serve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('action',choices=['doctor','run','status','stop']);p.add_argument('--run-dir',type=Path)
    args=p.parse_args();c=load(args.config)
    if args.action=='doctor':print(json.dumps(doctor(c),indent=2));return 0
    if args.run_dir is None:p.error('--run-dir required')
    if args.action=='run':return serve(c,args.run_dir)
    r=request_stop(args.run_dir) if args.action=='stop' else read(args.run_dir/'status.json')
    print(json.dumps(r,indent=2));return int(r['state']=='failed')


if __name__=='__main__':raise SystemExit(main())
