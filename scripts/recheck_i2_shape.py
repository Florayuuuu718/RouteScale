"""Read-only repeat audits; compare with frozen reports without overwriting them."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['performance','qualification','http','findings'],required=True)
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.kind=='findings':
        from scripts.audit_i2_numerical_findings import audit
        result=audit();path=ROOT/'results/i_minisglang/i2_numerical_trace_v1/independent_bundle_audit.json'
    else:
        if args.directory is None:raise ValueError('Directory required')
        if args.kind=='performance':from scripts.audit_i2_shape_performance import audit
        elif args.kind=='qualification':from scripts.audit_i2_shape_qualification import audit
        else:from scripts.audit_i2_shape_lifecycle import audit
        result=audit(args.directory);path=args.directory/('independent_audit.json' if args.kind=='http' else 'audit.json')
    if result!=json.loads(path.read_text()):raise ValueError('Recomputed audit differs from frozen evidence')
    print(args.kind,'frozen audit reproduced')


if __name__=='__main__':main()
