"""F launch/gate guards run in the small CPU CI without importing Torch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_f_ep import (CONFIG, ROOT, SOURCES, jobs, qualification_gate,
                                  require_idle_gpus, run_bounded, sha)


class FPreparationTests(unittest.TestCase):
    def test_matrix_distinguishes_local_processes_from_distinct_gpus(self):
        self.assertEqual(len(jobs()), 12)
        self.assertEqual({j["world"] for j in jobs() if j["device"] == "cpu"}, {1, 2, 4})
        self.assertEqual({j["world"] for j in jobs() if j["device"] == "cuda"}, {1})
        self.assertEqual({j["world"] for j in jobs(False)}, {1, 2})
        config = json.loads(CONFIG.read_text())
        self.assertFalse(config["paid_resources_authorized"])
        self.assertFalse(config["performance_execution_allowed"])

    def test_insufficient_or_busy_gpu_rejected_before_torch_launch(self):
        for inventory, message in (("0, 0, 0\n", "distinct GPUs"),
                                   ("0, 32, 0\n1, 0, 0\n", "idle"),
                                   ("0, 0, 0\n1, 0, 20\n", "idle")):
            with self.subTest(inventory=inventory), patch("scripts.prepare_f_ep.subprocess.check_output", return_value=inventory):
                with self.assertRaisesRegex(ValueError, message):
                    require_idle_gpus(2)

    def test_ep1_or_cpu_evidence_does_not_unlock_ep2(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                qualification_gate(tmp)

    def fixture(self, directory):
        for job in jobs(False):
            path = directory / job["label"]
            path.mkdir()
            raw = path / "raw.pt"
            raw.write_bytes(b"synthetic guard fixture, never a numerical acceptance")
            (path / "rank0.json").write_text("{}")
            manifest = {**job, "config_sha256": sha(CONFIG),
                        "source_sha256": {name: sha(ROOT / name) for name in SOURCES}}
            (path / "job.json").write_text(json.dumps(manifest))
            audit = {"passed": True, "job_sha256": sha(path / "job.json"),
                     "raw_sha256": {str(raw): sha(raw)}, "report_sha256": {"rank0.json": sha(path / "rank0.json")}}
            (path / "audit.json").write_text(json.dumps(audit))

    def test_failed_audit_and_changed_raw_or_reports_rejected(self):
        for mutation, message in (("audit", "failed"), ("raw", "tensors"), ("report", "reports")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                target = root / jobs(False)[0]["label"]
                if mutation == "audit":
                    path = target / "audit.json"
                    value = json.loads(path.read_text())
                    value["passed"] = False
                    path.write_text(json.dumps(value))
                elif mutation == "raw":
                    (target / "raw.pt").write_bytes(b"tampered")
                else:
                    (target / "rank0.json").write_text("tampered")
                with self.assertRaisesRegex(ValueError, message):
                    qualification_gate(root)

    def test_scope_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / jobs(False)[0]["label"] / "job.json"
            value = json.loads(path.read_text())
            value["world"] = 4
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "scope"):
                qualification_gate(root)

    def test_timeout_retains_log_and_exits_child_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeout.log"
            result = run_bounded([sys.executable, "-u", "-c", "import time; print('started'); time.sleep(60)"], path, 0.5)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("started", path.read_text())
            self.assertLess(result["wall_s"], 15)


if __name__ == "__main__":
    unittest.main()
