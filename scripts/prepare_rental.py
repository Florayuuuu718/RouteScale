"""Build/verify/restore a portable rental bundle or install its pinned environments."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rental.common import ROOT
from rental.deployment import build_bundle, verify_bundle, restore_bundle, install_environments, environment_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['package', 'verify', 'restore', 'install', 'check-env'])
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    root = args.root.absolute()
    bundle = args.bundle.absolute() if args.bundle else None
    if args.action in ('package', 'verify', 'restore') and bundle is None:
        parser.error('--bundle is required')
    if args.action == 'package':
        result = build_bundle(bundle, root)
        print({'files': len(result['files']), 'bytes': result['bytes'], 'commits': result['commits']})
    elif args.action == 'verify':
        result = verify_bundle(bundle)
        print({'passed': True, 'files': len(result['files']), 'bytes': result['bytes']})
    elif args.action == 'restore':
        print(restore_bundle(bundle, root))
    elif args.action == 'install':
        print({'passed': True, 'environments': list(install_environments(root, args.offline))})
    else:
        print({'passed': True, 'environments': list(environment_report(root))})


if __name__ == '__main__':
    main()
