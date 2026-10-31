import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from serving.h2_linear_policy import LINEAR_POLICY, install_prefill_linear_policy, validate_linear_policy


class RecordingLinear(nn.Linear):
    def __init__(self):
        super().__init__(16, 8, bias=True, dtype=torch.bfloat16)
        self.rows = []

    def forward(self, x):
        self.rows.append(x.shape[0])
        return super().forward(x)


class LinearPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.layer = RecordingLinear()
        self.context = SimpleNamespace(is_prefill=True)
        self.original = self.layer.forward
        self.weight = self.layer.weight.detach().clone()
        self.bias = self.layer.bias.detach().clone()
        self.restore, self.metadata = install_prefill_linear_policy(
            [self.layer], lambda: self.context, copy.deepcopy(LINEAR_POLICY))

    def tearDown(self):
        self.restore()

    def test_real_512_row_calls_without_padding_casting_or_weight_change(self):
        x = torch.randn(1024, 16, dtype=torch.bfloat16)
        out = self.layer(x)
        self.assertEqual(self.layer.rows, [512, 512])
        self.assertEqual(out.shape, (1024, 8))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(self.layer.weight, self.weight))
        self.assertTrue(torch.equal(self.layer.bias, self.bias))
        self.assertEqual(self.metadata["padding_rows"], 0)

    def test_batch8_and_two_batch4_calls_match_exactly(self):
        x = torch.randn(1024, 16, dtype=torch.bfloat16)
        full = self.layer(x)
        halves = torch.cat([self.layer(part) for part in x.split(512)])
        self.assertTrue(torch.equal(full, halves))

    def test_decode_does_not_change_native_call_shape(self):
        self.context.is_prefill = False
        self.layer(torch.randn(8, 16, dtype=torch.bfloat16))
        self.assertEqual(self.layer.rows, [8])

    def test_rejects_unsupported_prefill_rows_or_dtype(self):
        for shape, dtype in (((128, 16), torch.bfloat16), ((512, 16), torch.float32)):
            with self.assertRaisesRegex(ValueError, "outside"):
                self.layer(torch.randn(*shape, dtype=dtype))

    def test_restore_releases_bound_adapter_and_is_repeatable(self):
        self.restore()
        self.assertEqual(self.layer.forward, self.original)
        self.layer(torch.randn(1024, 16, dtype=torch.bfloat16))
        self.assertEqual(self.layer.rows, [1024])
        self.restore()

    def test_rejects_unqualified_policy_and_empty_selection(self):
        unknown = copy.deepcopy(LINEAR_POLICY)
        unknown["tile_rows"] = 256
        with self.assertRaises(ValueError):
            validate_linear_policy(unknown)
        with self.assertRaises(ValueError):
            install_prefill_linear_policy([], lambda: self.context, LINEAR_POLICY)
        with self.assertRaises(ValueError):
            install_prefill_linear_policy([self.layer], lambda: self.context, None)
