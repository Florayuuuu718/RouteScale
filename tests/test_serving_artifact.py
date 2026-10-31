from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from model import GPT, GPTConfig
from serving.artifact import (
    export_inference_artifact,
    load_inference_artifact,
    normalize_state_dict,
)


class ServingArtifactTests(unittest.TestCase):
    def test_normalize_state_dict_rejects_compile_prefix_collision(self) -> None:
        with self.assertRaisesRegex(ValueError, "collision"):
            normalize_state_dict(
                {
                    "weight": torch.ones(1),
                    "_orig_mod.weight": torch.ones(1),
                }
            )

    def test_export_excludes_training_state_and_strictly_reloads(self) -> None:
        torch.manual_seed(7)
        config = GPTConfig(
            block_size=8,
            vocab_size=50257,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        source_model = GPT(config).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "checkpoint.pt"
            artifact_path = root / "model.pt"
            manifest_path = root / "manifest.json"
            torch.save(
                {
                    "model": source_model.state_dict(),
                    "model_args": asdict(config),
                    "optimizer": {"must_not": "leak"},
                    "iter_num": 3,
                    "best_val_loss": torch.tensor(1.25),
                    "config": {"dtype": "bfloat16"},
                },
                checkpoint_path,
            )

            manifest = export_inference_artifact(
                checkpoint_path=checkpoint_path,
                artifact_path=artifact_path,
                manifest_path=manifest_path,
            )
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            self.assertNotIn("optimizer", artifact)
            self.assertTrue(manifest["tied_embedding_verified"])
            self.assertEqual(manifest["provenance"]["training_step"], 3)

            loaded_model, metadata = load_inference_artifact(artifact_path)
            inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)
            with torch.inference_mode():
                expected, _ = source_model(inputs)
                actual, _ = loaded_model(inputs)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(metadata["state_dict_sha256"], manifest["state_dict_sha256"])
            self.assertEqual(
                loaded_model.transformer.wte.weight.data_ptr(),
                loaded_model.lm_head.weight.data_ptr(),
            )


if __name__ == "__main__":
    unittest.main()

