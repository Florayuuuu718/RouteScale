"""Audit and export TP/F timing and trace evidence, without importing Torch."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rental.report import export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.directory.absolute(), args.output.absolute())
    print({key: result[key] for key in ('passed', 'multigpu_accepted', 'session_status')})
    if not result['passed']:
        raise SystemExit(1)
