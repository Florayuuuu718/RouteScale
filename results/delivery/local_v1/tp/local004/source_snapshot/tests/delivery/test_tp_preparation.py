from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from serving.delivery import write
from serving.tp_preparation import CONFIG,engine_rendezvous,jobs,load_plan,numerical_gate,require_gpus


class TPPreparationTests(unittest.TestCase):
    def test_explicit_engine_store_does_not_reuse_torchrun_agent_store(self):
        import os
        with patch.dict(os.environ,{'TORCHELASTIC_USE_AGENT_STORE':'True'}):
            self.assertEqual(engine_rendezvous(),'True')
            self.assertNotIn('TORCHELASTIC_USE_AGENT_STORE',os.environ)

    def test_complete_tp1234_matrix_is_separate_from_historical_gates(self):
        c=load_plan();q=jobs(c,'qualify');b=jobs(c,'benchmark')
        self.assertEqual(len(q),30);self.assertEqual(len(b),90)
        self.assertEqual(len({j['label'] for j in q}),30)
        self.assertEqual(len({j['label'] for j in b}),90)
        self.assertEqual({j['world'] for j in q},{1,2,4})
        self.assertFalse(c['paid_resources_authorized'])

    def test_unavailable_tp2_rejected_before_launch(self):
        with patch('serving.tp_preparation.subprocess.check_output',return_value='0, 0\n'):
            with self.assertRaisesRegex(ValueError,'2 GPUs'):require_gpus(2)

    def test_occupied_gpu_rejected(self):
        with patch('serving.tp_preparation.subprocess.check_output',return_value='0, 92\n'):
            with self.assertRaisesRegex(ValueError,'idle'):require_gpus(1)

    def test_local_probe_cannot_unlock_multigpu_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);c=load_plan()
            write(p/'qualification_gate.json',dict(passed=True,config_sha256='current',model_manifest_sha256='model',
                qualified_labels=[j['label'] for j in jobs(c,'qualify') if j['world']==1]))
            with patch('serving.tp_preparation.sha',side_effect=lambda path:'current' if path==CONFIG else 'model'):
                with self.assertRaisesRegex(ValueError,'Incomplete'):numerical_gate(p,c)

    def test_failed_gate_cannot_unlock_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'qualification_gate.json',dict(passed=False))
            with self.assertRaisesRegex(ValueError,'qualification'):numerical_gate(p,load_plan())

    def test_stale_gate_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'qualification_gate.json',dict(passed=True,config_sha256='old'))
            with self.assertRaisesRegex(ValueError,'qualification'):numerical_gate(p,load_plan())

    def test_changed_source_rejected(self):
        c=load_plan()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'qualification_gate.json',dict(passed=True,config_sha256='current',model_manifest_sha256='current',
                qualified_labels=[j['label'] for j in jobs(c,'qualify')],local_only=False,multigpu_accepted=True,
                source_sha256={'x.py':'old'},evidence_sha256={}))
            with patch('serving.tp_preparation.sha',return_value='current'):
                with self.assertRaisesRegex(ValueError,'source changed'):numerical_gate(p,c)


if __name__=='__main__':unittest.main()
