from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distributed_common import MemmapTokenDataset


class MemmapTokenDatasetTests(unittest.TestCase):
    def test_exact_window_and_shifted_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = np.arange(21, dtype=np.uint16)
            validation = np.arange(101, 118, dtype=np.uint16)
            train.tofile(root / "train.bin")
            validation.tofile(root / "val.bin")
            manifest = {"vocab_size": 128, "splits": {}}
            (root / "manifest.json").write_text(json.dumps(manifest))

            dataset = MemmapTokenDataset(root, block_size=4)
            self.assertEqual(dataset.train_window_count, 5)
            self.assertEqual(dataset.val_window_count, 4)
            x, y = dataset.batch("train", [0, 3], "cpu")
            np.testing.assert_array_equal(x.numpy(), [[0, 1, 2, 3], [12, 13, 14, 15]])
            np.testing.assert_array_equal(y.numpy(), [[1, 2, 3, 4], [13, 14, 15, 16]])
            self.assertEqual(dataset.vocab_size, 128)
            expected = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            self.assertEqual(dataset.manifest_sha256, expected)

    def test_fixed_validation_ids_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.arange(21, dtype=np.uint16).tofile(root / "train.bin")
            np.arange(21, dtype=np.uint16).tofile(root / "val.bin")
            (root / "manifest.json").write_text(
                json.dumps({"vocab_size": 32, "splits": {}})
            )
            dataset = MemmapTokenDataset(root, block_size=4)
            first = dataset.fixed_validation_ids(4, 123)
            second = dataset.fixed_validation_ids(4, 123)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(set(first.tolist())), 4)


if __name__ == "__main__":
    unittest.main()
