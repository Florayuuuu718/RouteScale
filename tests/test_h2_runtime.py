import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from serving.h2_prefix_probe import prefix_disabled_manager
from serving.h2_runtime import aggregate_waves, assert_released, check_capacity, quantile
from scripts.run_h2_nanovllm import pilot_jobs


class I2RuntimeTests(unittest.TestCase):
    def test_current_b8_pilot_does_not_silently_include_failed_batch_one(self):
        root = Path(__file__).resolve().parents[1]
        pilot = json.loads((root / "config/serving/h2_graph_b8.json").read_text())
        config = json.loads((root / pilot["h2_config"]).read_text())
        selected = {**config, "pilot_case_ids": pilot["pilot_case_ids"]}
        jobs = list(pilot_jobs(selected, pilot["variants"]))
        self.assertEqual(len(jobs), 18)
        self.assertEqual(set(pilot["pilot_case_ids"]), set(pilot["correctness_case_ids"]))
        self.assertTrue(all(case.startswith("b8_") for case, _, _ in jobs))
        self.assertTrue(all(name != "eager_prefix" for _, name, _ in jobs))

    def test_capacity_accounts_for_decode_not_just_prefill(self):
        engine = dict(num_kvcache_blocks=24, kvcache_block_size=256,
            max_num_seqs=8, max_num_batched_tokens=4096, max_model_len=1024)
        case = dict(batch_size=8, prompt_length=512, output_length=64)
        self.assertEqual(check_capacity(engine, case), 24)
        with self.assertRaisesRegex(ValueError, "Cold decode"):
            check_capacity({**engine, "num_kvcache_blocks": 23}, case)
        for key, value in (("max_num_seqs", 7), ("max_num_batched_tokens", 4095), ("max_model_len", 575)):
            with self.assertRaises(ValueError):
                check_capacity({**engine, key: value}, case)

    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(quantile([4, 1, 2, 3], .5), 2.5)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], .95), 3.85)
        with self.assertRaises(ValueError):
            quantile([], .5)

    def test_aggregation_uses_sum_of_wave_durations(self):
        waves = []
        for seconds in (1, 3):
            waves.append(dict(wall_seconds=seconds, peak_allocated_bytes=100 * seconds,
                cached_prompt_tokens=256, requests=[dict(ttft_ms=10 * seconds,
                    tpot_ms=20 * seconds, e2e_ms=30 * seconds, prompt_tokens=512, output_tokens=64)]))
        result = aggregate_waves(waves)
        self.assertEqual(result["output_tokens_per_second"], 128 / 4)
        self.assertEqual(result["ttft_p50_ms"], 20)
        self.assertEqual(result["peak_allocated_bytes"], 300)
        self.assertEqual(result["cached_prompt_tokens"], 512)
        self.assertEqual(result["requests_completed"], 2)

    def test_no_prefix_switch_does_not_modify_publication_or_release(self):
        class Manager:
            def publish(self):
                return "unchanged"
        manager = prefix_disabled_manager(Manager)()
        manager.free_block_ids = [1, 2]
        self.assertEqual(manager.can_allocate(SimpleNamespace(num_blocks=2)), 0)
        self.assertEqual(manager.can_allocate(SimpleNamespace(num_blocks=3)), -1)
        self.assertEqual(manager.publish(), "unchanged")

    def test_release_check_catches_refcount_and_queue_leaks(self):
        manager = SimpleNamespace(used_block_ids=set(), free_block_ids=[0, 1],
            blocks=[SimpleNamespace(ref_count=0), SimpleNamespace(ref_count=0)])
        llm = SimpleNamespace(scheduler=SimpleNamespace(block_manager=manager),
            is_finished=lambda: True, _h2_num_blocks=2)
        assert_released(llm)
        manager.blocks[0].ref_count = 1
        with self.assertRaisesRegex(AssertionError, "leaked"):
            assert_released(llm)
        manager.blocks[0].ref_count = 0
        llm.is_finished = lambda: False
        with self.assertRaises(AssertionError):
            assert_released(llm)

    def test_pilot_is_bounded_and_variant_order_rotates(self):
        config = dict(independent_runs=3, pilot_case_ids=["a", "b", "c", "d"],
            variants={"eager_no_prefix": {}, "graph_no_prefix": {}, "eager_prefix": {}})
        jobs = list(pilot_jobs(config))
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(set(jobs)), 36)
        self.assertEqual([jobs[i][1] for i in (0, 12, 24)],
            ["eager_no_prefix", "graph_no_prefix", "eager_prefix"])
        graph = list(pilot_jobs(config, ["eager_no_prefix", "graph_no_prefix"]))
        self.assertEqual(len(graph), 24)
        self.assertEqual(len(set(graph)), 24)
        self.assertEqual([graph[i][1] for i in (0, 8, 16)],
            ["eager_no_prefix", "graph_no_prefix", "eager_no_prefix"])


if __name__ == "__main__":
    unittest.main()
