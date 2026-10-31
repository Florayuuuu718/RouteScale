"""Exercise numerical-audit rejection using temporary copies, preserving evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    from scripts.audit_i2_external_alignment import audit
    from serving.i2_workloads import digest
    output = args.directory.parent / 'audit_negative_controls.json'
    if output.exists():
        raise ValueError('Preserve audit controls')
    frozen = json.loads((args.directory / 'audit.json').read_text())
    result = audit(args.directory)
    if result != frozen:
        raise ValueError('Positive audit does not reproduce')
    rows = []
    for name, expected in [('threshold', 'unchanged_threshold'), ('metric', ':metrics'), ('source', 'snapshot:')]:
        with tempfile.TemporaryDirectory(prefix='i2_external_audit_', dir=ROOT / 'runs/tmp') as temporary:
            copy = Path(temporary) / 'evidence'
            shutil.copytree(args.directory, copy)
            if name == 'threshold':
                path = copy / 'provenance.json'
                value = json.loads(path.read_text())
                value['config']['numerics']['max_relative_rmse'] = .01
                path.write_text(json.dumps(value))
            elif name == 'metric':
                path = copy / 'native_fp16.json'
                value = json.loads(path.read_text())
                value['rows'][0]['errors'][0]['relative_rmse'] += .01
                path.write_text(json.dumps(value))
            else:
                path = copy / 'source_snapshot/serving/i2_external_alignment.py'
                path.write_text(path.read_text() + '\n# altered\n')
            try:
                audit(copy)
            except ValueError as error:
                reason = str(error)
                if expected not in reason:
                    raise
                rows.append(dict(control=name, rejected=True, reason=reason))
            else:
                raise ValueError('Audit accepted corruption: ' + name)
    record = dict(passed=True, positive_audit_reproduced=True, negative_controls=rows,
        base_directory=str(args.directory.relative_to(ROOT) if args.directory.is_absolute() else args.directory),
        base_audit_sha256=digest(args.directory / 'audit.json'),
        source_sha256={name: digest(ROOT/name) for name in
            ('scripts/check_i2_external_audit.py', 'scripts/audit_i2_external_alignment.py')},
        performance_allowed=False)
    output.write_text(json.dumps(record, indent=2) + '\n')
    snapshot = output.parent / 'audit_control_source_snapshot/scripts'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in record['source_sha256']:
        shutil.copy2(ROOT/name, snapshot / Path(name).name)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
