import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest
from scripts.run_i2_shape_performance import prerequisites


class TimingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.q=self.root/'q';self.l=self.root/'l'
        self.q.mkdir();self.l.mkdir();model=self.root/'model';model.mkdir();(model/'weight').write_bytes(b'known')
        self.config=json.loads((ROOT/'config/serving/i2_single_gpu.json').read_text())
        self.config.update(model_path=str(model),policy='fi_prefill_no_split_linear512')
        self.config['numerics']['reference']='independent_shape_controlled_naive_BF16'
        self.spec=dict(qualification_directory=str(self.q),lifecycle_directory=str(self.l),reference_scope='within_native_BF16_strategy_invariance')
        names=['serving/i2_runtime.py','serving/i2_worker.py','serving/i2_numerical_policy.py','serving/i2_linear_tile_policy.py']
        self.sources={n:digest(ROOT/n) for n in names}
        self.write(self.q/'audit.json',dict(integrity_passed=True,qualification_passed=True,compared_vectors=12544,reference_scope=self.spec['reference_scope'],source_sha256=digest(ROOT/'scripts/audit_i2_shape_qualification.py')))
        self.write(self.q/'provenance.json',dict(config=self.config,source_sha256=self.sources,model_sha256={'weight':digest(model/'weight')},reference_scope=self.spec['reference_scope']))
        self.workload={'cases':[]};self.write(self.q/'workload.json',dict(suites=[{'name':'control512','workload':self.workload}]))
        self.write(self.l/'provenance.json',dict(config=self.config,source_sha256=self.sources,qualification_sha256=digest(self.q/'audit.json')))
        self.write(self.l/'summary.json',dict(HTTP_lifecycle_passed=True))
        self.write(self.l/'independent_audit.json',dict(passed=True,provenance_sha256=digest(self.l/'provenance.json'),source_sha256=digest(ROOT/'scripts/audit_i2_shape_lifecycle.py')))
        for name in self.config['variants']:
            folder=self.l/f'service_{name}';folder.mkdir()
            self.write(folder/'audit.json',{'passed':True})
            self.write(folder/'events.json',[dict(event='i2_shape_policy',name=self.config['policy'],source_sha256={n:self.sources[n] for n in names[-2:]})])
        self.mock=patch('scripts.run_i2_service_gate.audit_service',return_value={'passed':True})
        self.mock.start();self.addCleanup(self.mock.stop)

    def write(self,path,value):path.write_text(json.dumps(value))

    def test_numerical_and_HTTP_scope_can_admit_only_the_qualified_subset(self):
        self.assertTrue(prerequisites(self.spec,self.config,self.workload)['performance_allowed'])
        with self.assertRaises(ValueError):prerequisites(self.spec,self.config,{'cases':['unqualified']})

    def test_missing_actual_engine_policy_marker_blocks_timing(self):
        self.write(self.l/'service_radix_overlap/events.json',[])
        with self.assertRaises(ValueError):prerequisites(self.spec,self.config,self.workload)

    def test_relaxed_tolerance_or_changed_checkpoint_blocks_timing(self):
        changed=copy.deepcopy(self.config);changed['numerics']['rtol']=.2
        with self.assertRaises(ValueError):prerequisites(self.spec,changed,self.workload)
        (Path(self.config['model_path'])/'weight').write_bytes(b'changed')
        with self.assertRaises(ValueError):prerequisites(self.spec,self.config,self.workload)

    def test_failed_independent_HTTP_audit_blocks_timing(self):
        self.write(self.l/'independent_audit.json',dict(passed=False,provenance_sha256=digest(self.l/'provenance.json')))
        with self.assertRaises(ValueError):prerequisites(self.spec,self.config,self.workload)


if __name__=='__main__':unittest.main()
