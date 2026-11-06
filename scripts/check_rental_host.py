"""Bounded host preflight subprocess used by the rental session controller."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rental.common import read
from rental.session import preflight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    preflight(read(args.directory / 'host_contract.json'), args.directory)
