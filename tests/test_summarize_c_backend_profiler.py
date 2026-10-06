from __future__ import annotations

import unittest

from scripts.summarize_c_backend_profiler import collective_family


class CollectiveFamilyTests(unittest.TestCase):
    def test_classifies_common_wrapper_and_kernel_names(self) -> None:
        cases = {
            "nccl:all_reduce": "all_reduce",
            "ncclDevKernel_AllReduce_RING_LL": "all_reduce",
            "c10d::reduce_scatter_tensor_": "reduce_scatter",
            "ncclDevKernel_ReduceScatter": "reduce_scatter",
            "c10d::allgather_into_tensor_coalesced_": "all_gather",
            "ncclDevKernel_AllGather": "all_gather",
            "nccl:barrier": "other_collective",
            "aten::mm": None,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(collective_family(name), expected)


if __name__ == "__main__":
    unittest.main()
