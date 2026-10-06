from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.package_c_results import include_file


class IncludeFileTests(unittest.TestCase):
    def test_excludes_checkpoints_traces_and_binary_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                root / "summary.json": True,
                root / "logs" / "run.log": True,
                root / "traces" / "rank0.json": False,
                root / "profile_rank0_trace.json": False,
                root / "checkpoints" / "model.json": False,
                root / "train.bin": False,
            }
            for path, expected in cases.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
                with self.subTest(path=path):
                    self.assertEqual(
                        include_file(path, root, 1024)[0], expected
                    )


if __name__ == "__main__":
    unittest.main()
