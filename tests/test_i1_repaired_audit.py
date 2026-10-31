"""Tamper controls: the auditor must reject changed evidence, not just bad status flags."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.audit_i1_repaired import audit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/i_minisglang/i1_repaired/accept002_healthy"


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (EVIDENCE / "audit.json").is_file():
            raise unittest.SkipTest("Real GPU evidence required for tamper controls")
        run = json.loads((EVIDENCE / "run.json").read_text())
        if not (ROOT / run["runtime_dir"]).is_dir():
            raise unittest.SkipTest("Carry the frozen raw logs to rerun GPU evidence tamper controls")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / "evidence"
        shutil.copytree(EVIDENCE, self.directory)
        self.addCleanup(self.tmp.cleanup)

    def mutate(self, name, change):
        path = self.directory / name
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))
        result = audit(self.directory)
        self.assertFalse(result["lifecycle_complete"])
        return result["failed_checks"]

    def test_real_healthy_evidence_passes(self):
        self.assertTrue(audit(self.directory)["lifecycle_complete"])

    def test_extra_client_terminal_is_detected(self):
        def change(data):
            data[0]["events"].append(copy.deepcopy(data[0]["events"][-1]))
        failed = self.mutate("requests.json", change)
        self.assertIn("every_offered_request_has_one_terminal", failed)
        self.assertIn("client_events_match_raw", failed)

    def test_over_capacity_event_is_detected(self):
        def change(data):
            row = next(r for r in data if r["event"] == "frontend_new_user")
            row["admitted_uids"] = list(range(5))
        failed = self.mutate("server_events.json", change)
        self.assertIn("admission_always_bounded", failed)
        self.assertIn("server_events_match_raw", failed)

    def test_frozen_source_tampering_is_detected(self):
        path = self.directory / "source_snapshot/serving/minisgl_observer.py"
        path.write_text(path.read_text() + "\n# tampered\n")
        self.assertIn("frozen_sources", audit(self.directory)["failed_checks"])

    def test_internal_shutdown_escalation_is_detected(self):
        def change(data):
            row = next(r for r in data if r["event"] == "workers_stopped")
            row["escalated"] = True
        self.assertIn("workers_stopped_without_internal_escalation", self.mutate("server_events.json", change))


if __name__ == "__main__":
    unittest.main()
