from __future__ import annotations

import unittest

from scripts.summarize_c_checkpoints import checkpoint_file_breakdown


class CheckpointFileBreakdownTests(unittest.TestCase):
    def test_classifies_deepspeed_and_dcp_files(self) -> None:
        result = checkpoint_file_breakdown(
            [
                "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
                "zero_pp_rank_0_mp_rank_00_model_states.pt",
                "__0_0.distcp",
                "rng_rank0.pt",
                ".metadata",
            ]
        )
        self.assertEqual(
            result,
            {
                "optimizer_state_files": 1,
                "model_state_files": 1,
                "distributed_checkpoint_shards": 1,
                "rng_state_files": 1,
                "other_files": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
