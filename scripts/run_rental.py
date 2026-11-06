"""Plan without GPU access, or execute sequentially on a declared existing host."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rental.common import read, write
from rental.session import execute, plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host-config', type=Path)
    parser.add_argument('--resume', action='store_true', help='Continue a successfully completed phase; retain original deadline')
    parser.add_argument('--through', choices=['qualification', 'performance', 'trace'], default='trace')
    args = parser.parse_args()
    if not args.execute:
        args.directory.mkdir(parents=True, exist_ok=False)
        result = plan(args.directory)
        write(args.directory / 'plan.json', result)
        print(result['counts'])
    else:
        if not args.host_config:
            parser.error('--execute requires --host-config')
        result = execute(args.directory, read(args.host_config), args.through, args.resume)
        print({'passed': result['passed'], 'status': result['status'], 'cloud_billing_stopped': False})
        if not result['passed']:
            raise SystemExit(1)
