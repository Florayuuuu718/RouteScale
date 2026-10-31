import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from serving.h2_numerical_policy import install_shape_policy
from serving.verify_h2_corrected import compare_reports


class ShapePolicyTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(block_tables=None)
        module = ModuleType("nanovllm.utils.context")
        module.get_context = lambda: self.context
        self.patch = patch.dict(sys.modules, {"nanovllm.utils.context": module})
        self.patch.start()

        context = self.context
        class Runner:
            config = SimpleNamespace(max_model_len=1024)
            block_size = 256
            def prepare_block_tables(self, seqs):
                width = max(len(s.block_table) for s in seqs)
                return torch.tensor([s.block_table + [-1] * (width-len(s.block_table)) for s in seqs], dtype=torch.int32)
            def prepare_prefill(self, seqs):
                context.block_tables = None
                return "inputs", "positions"
        self.runner = Runner()

    def tearDown(self):
        self.patch.stop()

    def test_static_width_preserves_physical_page_ids_and_dtype(self):
        restore = install_shape_policy(self.runner, static_tables=True)
        tables = self.runner.prepare_block_tables([SimpleNamespace(block_table=[5]), SimpleNamespace(block_table=[7, 3])])
        self.assertEqual(tables.tolist(), [[5, -1, 0, 0], [7, 3, 0, 0]])
        self.assertEqual(tables.dtype, torch.int32)
        restore()

    def test_rejects_tables_that_exceed_model_capacity(self):
        install_shape_policy(self.runner, static_tables=True)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.runner.prepare_block_tables([SimpleNamespace(block_table=[1, 2, 3, 4, 5])])

    def test_paged_prefill_uses_current_kv_tables_even_with_no_hit(self):
        restore = install_shape_policy(self.runner, static_tables=True, paged_prefill=True)
        self.assertEqual(self.runner.prepare_prefill([SimpleNamespace(block_table=[6, 2])]), ("inputs", "positions"))
        self.assertEqual(self.context.block_tables.tolist(), [[6, 2, 0, 0]])
        restore()

    def test_initial_model_warmup_without_cache_keeps_original_path(self):
        install_shape_policy(self.runner, static_tables=True, paged_prefill=True)
        self.runner.prepare_prefill([SimpleNamespace(block_table=[])])
        self.assertIsNone(self.context.block_tables)

    def test_restore_removes_both_interventions(self):
        original_tables, original_prefill = self.runner.prepare_block_tables, self.runner.prepare_prefill
        restore = install_shape_policy(self.runner, static_tables=True, paged_prefill=True)
        restore()
        self.assertEqual(self.runner.prepare_block_tables, original_tables)
        self.assertEqual(self.runner.prepare_prefill, original_prefill)


def fixtures():
    reports = {}
    for name, eager, prefix in (("eager_no_prefix", True, False), ("graph_no_prefix", False, False), ("eager_prefix", True, True)):
        phases = {}
        for phase in ("cold", "warm"):
            phases[phase] = {"cached_tokens_by_request": [256 if prefix and phase == "warm" else 0],
                "output_tokens": [64], "released_state": {"running": 0, "waiting": 0, "used_blocks": 0},
                "logits": [{"step": index, "is_prefill": index == 0, "shape": [1, 10], "dtype": "bfloat16",
                    "sha256": "a" * 64, "top1_token_ids": [0]} for index in range(64)]}
        reports[name] = {"status": "pass", "variant": name, "evidence_class": "functional_gpu_policy_validation",
            "provenance": {"only": "unit-test fixture"},
            "engine": {"dtype": "bfloat16", "num_kvcache_blocks": 32, "enforce_eager": eager, "prefix_cache": prefix},
            "cases": {"fixture": {"workload_sha256": "fixture", "batch_size": 1,
                "expected_warm_cached_tokens_per_request": 256, "phases": phases}}}
    return reports


class CorrectedComparisonTests(unittest.TestCase):
    def test_accepts_only_complete_exact_three_path_evidence(self):
        rows = compare_reports(fixtures(), ["fixture"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["exact"] for row in rows))

    def test_detects_one_changed_full_vocabulary_logit_hash(self):
        reports = fixtures()
        reports["graph_no_prefix"]["cases"]["fixture"]["phases"]["cold"]["logits"][31]["sha256"] = "b" * 64
        row = next(r for r in compare_reports(reports, ["fixture"]) if r["variant"] == "graph_no_prefix" and r["phase"] == "cold")
        self.assertFalse(row["exact"])
        self.assertEqual(row["mismatched_steps"], [31])

    def test_rejects_missing_steps_or_variants(self):
        reports = fixtures()
        del reports["eager_prefix"]
        with self.assertRaisesRegex(ValueError, "three"):
            compare_reports(reports, ["fixture"])
        reports = fixtures()
        reports["graph_no_prefix"]["cases"]["fixture"]["phases"]["cold"]["logits"].pop()
        with self.assertRaisesRegex(ValueError, "64"):
            compare_reports(reports, ["fixture"])

    def test_rejects_wrong_label_pool_and_provenance(self):
        for key, value in (("variant", "eager_prefix"), ("engine", {"dtype": "float32"}), ("provenance", {"changed": True})):
            reports = fixtures()
            reports["graph_no_prefix"][key] = value
            with self.assertRaises(ValueError):
                compare_reports(reports, ["fixture"])
        reports = fixtures()
        reports["graph_no_prefix"]["engine"]["num_kvcache_blocks"] = 16
        with self.assertRaisesRegex(ValueError, "pool"):
            compare_reports(reports, ["fixture"])

    def test_rejects_unexpected_hits_or_unreleased_state(self):
        for field, value in (("cached_tokens_by_request", [256]), ("output_tokens", [63]), ("released_state", {"used": 1})):
            reports = fixtures()
            reports["eager_no_prefix"]["cases"]["fixture"]["phases"]["cold"][field] = value
            with self.assertRaises(ValueError):
                compare_reports(reports, ["fixture"])


if __name__ == "__main__":
    unittest.main()
