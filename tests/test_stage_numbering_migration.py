import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.migrate_stage_numbering import (
    MANIFEST, apply_migration, check_migration, rename_identifiers, translate_text,
)


class StageNumberingMigrationTests(unittest.TestCase):
    def test_translates_modules_fields_and_stage_names_without_changing_terms(self):
        text = "serving.i2_runtime _i2_cache required_i1_results H2 H2D H100 Q/K/V I/J 阶段"
        self.assertEqual(
            translate_text(text, "serving/i3_model.py"),
            "serving.h2_runtime _h2_cache required_h1_results G2 H2D H100 Q/K/V H/I 阶段",
        )
        self.assertEqual(rename_identifiers("docs/h_stage_protocol.md"), "docs/g_stage_protocol.md")
        self.assertEqual(rename_identifiers("docs/i_stage_protocol.md"), "docs/h_stage_protocol.md")
        self.assertEqual(rename_identifiers("results/j_minisglang/j1_lifecycle"), "results/i_minisglang/i1_lifecycle")

    def test_current_roadmap_labels_and_unrelated_hypotheses_are_preserved(self):
        self.assertEqual(
            translate_text("H3: config/serving/i3_routescale.json", "README.md"),
            "H3: config/serving/h3_routescale.json",
        )
        self.assertEqual(translate_text("H1 H2", "docs/RouteScale_E阶段项目复盘与面试指南.md"), "H1 H2")

    def test_migrates_colliding_names_and_checksum_chain_with_original_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/tmp").mkdir(parents=True)
            (root / "runs/downloads/i1_wheel_parts").mkdir(parents=True)

            def write(path, data):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

            write("README.md", b"H3 current roadmap\n")
            write("runs/README.md", b"Local workspace\n")
            write("docs/h_stage_protocol.md", b"H-stage native inference\n")
            write("docs/i_stage_protocol.md", b"I-stage nano-vLLM\n")
            config = b'{"stage":"I1","tolerance":0.005}\n'
            write("config/serving/i1_nanovllm.json", config)
            raw = json.dumps(dict(stage="I1", config_sha256=hashlib.sha256(config).hexdigest(),
                                  status="fail", seconds=123.456, requests=11)).encode() + b"\n"
            write("results/i_nanovllm/i1_run.json", raw)
            write("results/i_nanovllm/audit.json", json.dumps(dict(raw_sha256=hashlib.sha256(raw).hexdigest())).encode())
            binary = b"unchanged H1 I3 i2_tensor\x00\xff"
            write("runs/validation/i3_routescale/reference.pt", binary)
            apply_migration(root)
            check_migration(root)
            self.assertEqual((root / "docs/g_stage_protocol.md").read_text(), "G-stage native inference\n")
            self.assertEqual((root / "docs/h_stage_protocol.md").read_text(), "H-stage nano-vLLM\n")
            new_config = (root / "config/serving/h1_nanovllm.json").read_bytes()
            new_raw_bytes = (root / "results/h_nanovllm/h1_run.json").read_bytes()
            new_raw = json.loads(new_raw_bytes)
            self.assertEqual(new_raw["config_sha256"], hashlib.sha256(new_config).hexdigest())
            self.assertEqual(new_raw["stage"], "H1")
            self.assertEqual((new_raw["status"], new_raw["seconds"], new_raw["requests"]), ("fail", 123.456, 11))
            audit = json.loads((root / "results/h_nanovllm/audit.json").read_text())
            self.assertEqual(audit["raw_sha256"], hashlib.sha256(new_raw_bytes).hexdigest())
            self.assertEqual((root / "runs/validation/h3_routescale/reference.pt").read_bytes(), binary)
            self.assertFalse((root / "results/i_nanovllm").exists())
            self.assertTrue((root / "runs/downloads/h1_wheel_parts").is_dir())
            self.assertFalse((root / "runs/downloads/i1_wheel_parts").exists())
            self.assertTrue((root / MANIFEST).exists())
            with self.assertRaisesRegex(ValueError, "already recorded"):
                apply_migration(root)
            (root / "results/h_nanovllm/h1_run.json").write_bytes(new_raw_bytes.replace(b"123.456", b"123.457"))
            with self.assertRaisesRegex(ValueError, "Migrated file changed"):
                check_migration(root)


if __name__ == "__main__":
    unittest.main()
