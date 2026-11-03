"""Full-training resource and qualification guards; no Torch dependency."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_f_training import jobs, performance_gate


class FTrainingPreparationTests(unittest.TestCase):
    def test_full_local_and_rental_matrices(self):
        self.assertEqual(len(jobs()), 4)
        self.assertEqual(len(jobs(host=True)), 5)
        matrix = jobs(benchmark=True)
        self.assertEqual(len(matrix), 9)
        self.assertEqual({j["mode"] for j in matrix}, {"reference", "replicated", "ep"})
        self.assertTrue(all(j["world"] == (1 if j["mode"] == "reference" else 2) for j in matrix))

    def test_missing_multigpu_module_gate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                performance_gate(tmp)

    def test_different_module_host_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "module/cuda1_float32"
            path.mkdir(parents=True)
            (path / "job.json").write_text(json.dumps({"host": "another-host"}))
            with patch("scripts.prepare_f_training.qualification_gate", return_value=True):
                with self.assertRaisesRegex(ValueError, "same host"):
                    performance_gate(root)

    def test_local_full_shape_alone_cannot_unlock_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training/cuda1_full").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                performance_gate(root)


if __name__ == "__main__":
    unittest.main()
