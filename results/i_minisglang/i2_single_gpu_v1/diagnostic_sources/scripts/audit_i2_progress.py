"""Validate the preserved I2 failed-gate and lifecycle evidence, not acceptance."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest
from scripts.audit_i2_numerics import audit_numerics
from scripts.audit_i2_lifecycle_diagnostic import audit as audit_lifecycle


def audit():
    directory=ROOT/'results/i_minisglang/i2_single_gpu_v1'
    stored=json.loads((directory/'attempt001/numerical_gate.json').read_text())
    actual=audit_numerics(directory/'attempt001')
    # The failed audit source is frozen separately. The current auditor adds
    # explicit nonfinite handling; this dataset is finite. Compare all results.
    actual_source=actual.pop('source_sha256')
    stored_source=stored.pop('source_sha256')
    lifecycle=audit_lifecycle(directory/'lifecycle_diag001')
    cpu=json.loads((directory/'cpu_tests.json').read_text())
    manifest=json.loads((directory/'diagnostic_manifest.json').read_text())
    checks={
        'failed_audit_source':digest(directory/'attempt001/audit_source_snapshot/scripts/audit_i2_numerics.py')==stored_source,
        'independent_failed_numeric_result':actual==stored and actual['passed'] is False and actual['performance_allowed'] is False,
        'all_first_variant_vectors_checked':actual['compared_vectors']==1024 and len(actual['failures'])==1024,
        'all_first_variant_vectors_finite':all(r['finite'] for r in actual['failures']),
        'remaining_numerics_unrun':len(actual['missing_variants'])==6,
        'HTTP_lifecycle':lifecycle['passed'] and lifecycle==json.loads((directory/'lifecycle_diag001/independent_audit.json').read_text()),
        'CPU_tests':cpu['returncode']==0 and cpu['tests_passed']==12 and all(digest(ROOT/n)==value for n,value in cpu['source_sha256'].items()) and digest(directory/cpu['log'])==cpu['log_sha256'],
        'diagnostic_evidence_hashes':all(digest(directory/n)==value for n,value in manifest['evidence_sha256'].items()),
        'diagnostic_source_snapshots':all(digest(directory/'diagnostic_sources'/n)==value for n,value in manifest['source_sha256'].items()),
    }
    diagnostics=['native_strategy_diagnostic.json','rounding_diagnostic.json','kernel_diagnostic.json',
        'linear_shape_diagnostic.json','reference_precision_diagnostic.json']
    checks['no_diagnostic_unlocks_timing']=all(json.loads((directory/n).read_text())['performance_allowed'] is False for n in diagnostics)
    processes=[r for path in directory.glob('*/processes.json') for r in json.loads(path.read_text())]
    checks['no_formal_benchmark_processes']=not any(r['label'].startswith('benchmark') for r in processes)
    checks['dedicated_GPU_processes_closed']=all(r.get('gpu_after')=='0 MiB' for r in processes)
    return dict(stage='I2',integrity_passed=all(checks.values()),checks=checks,
        local_single_gpu_complete=False,numerical_passed=False,performance_allowed=False,
        formal_performance_processes=0,planned_performance_processes=21,
        HTTP_lifecycle_passed=lifecycle['passed'],HTTP_requests=lifecycle['requests'],HTTP_checks=lifecycle['scenario_checks'],
        CPU_tests_passed=cpu['tests_passed'],paid_resources_used=False,
        numerical_auditor_sha256=actual_source,source_sha256=digest(Path(__file__)))


if __name__=='__main__':
    result=audit()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['integrity_passed'] else 1)
