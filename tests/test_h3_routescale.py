from functools import partial
from types import SimpleNamespace
import unittest

import torch

from model import GPT,GPTConfig
from serving.h3_model import PackedRouteScaleGPT,SegmentedSDPA,validate_request
from serving.h3_prepare import byte_symbols,prepared_bpe


class I3RouteScaleTests(unittest.TestCase):
    def native(self,bias=False):
        torch.manual_seed(13)
        return GPT(GPTConfig(block_size=32,vocab_size=64,n_layer=2,n_head=2,n_embd=16,bias=bias,dropout=0)).eval()

    def packed(self,native,lengths):
        bounds=[0]
        for n in lengths: bounds.append(bounds[-1]+n)
        context=SimpleNamespace(is_prefill=True,cu_seqlens_q=torch.tensor(bounds))
        getter=lambda:context
        return PackedRouteScaleGPT(native,partial(SegmentedSDPA,get_context=getter),getter),context

    def test_packed_prefill_matches_native_equal_and_ragged(self):
        for bias in (False,True):
            native=self.native(bias)
            for lengths in ([7,7],[3,11,5]):
                model,_=self.packed(native,lengths)
                prompts=[torch.randint(0,64,(n,)) for n in lengths]
                tokens=torch.cat(prompts)
                positions=torch.cat([torch.arange(n) for n in lengths])
                with torch.inference_mode():
                    actual=model.compute_logits(model(tokens,positions))
                    expected=torch.cat([native(p.unsqueeze(0))[0][:,-1] for p in prompts])
                torch.testing.assert_close(actual,expected,atol=3e-6,rtol=1e-5)

    def test_original_weight_parameters_and_tying_are_retained(self):
        native=self.native()
        model,_=self.packed(native,[4])
        self.assertIs(model.gpt,native)
        self.assertEqual(set(model.state_dict()),{f"gpt.{k}" for k in native.state_dict()})
        self.assertIs(model.gpt.lm_head.weight,model.gpt.transformer.wte.weight)
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in native.parameters()))

    def test_absolute_positions_are_not_silently_reset_in_packed_rows(self):
        native=self.native()
        model,_=self.packed(native,[5,5])
        ids=torch.arange(10)
        correct=torch.arange(5).repeat(2)
        with torch.inference_mode():
            left=model.compute_logits(model(ids,correct))
            wrong=model.compute_logits(model(ids,torch.zeros(10,dtype=torch.long)))
        self.assertFalse(torch.allclose(left,wrong))

    def test_training_model_and_wrong_shapes_fail(self):
        native=self.native().train()
        with self.assertRaises(ValueError): self.packed(native,[4])
        model,_=self.packed(native.eval(),[4])
        with self.assertRaises(ValueError): model(torch.zeros(1,4,dtype=torch.long),torch.arange(4))

    def test_segmented_reference_cannot_be_used_for_decode(self):
        model,context=self.packed(self.native(),[4])
        context.is_prefill=False
        with self.assertRaises(ValueError): model(torch.arange(4),torch.arange(4))

    def test_final_sample_is_not_counted_as_another_cached_position(self):
        validate_request([1]*512,SimpleNamespace(max_tokens=1,temperature=.6))
        validate_request([1]*505,SimpleNamespace(max_tokens=8,temperature=.6))
        with self.assertRaises(ValueError): validate_request([1]*512,SimpleNamespace(max_tokens=2,temperature=.6))

    def test_invalid_ids_budget_and_temperature_rejected_before_enqueue(self):
        for ids,n,t in [([],8,.6),([50257],8,.6),([-1],8,.6),([True],8,.6),([1],0,.6),
                ([1],1.5,.6),([1],True,.6),([1],8,0),([1],8,float("nan"))]:
            with self.assertRaises(ValueError): validate_request(ids,SimpleNamespace(max_tokens=n,temperature=t))

    def test_byte_symbol_map_is_invertible_and_known_space_matches_gpt2(self):
        symbols=byte_symbols()
        self.assertEqual(len(symbols),256)
        self.assertEqual(len(set(symbols.values())),256)
        self.assertEqual(symbols[32],"Ġ")
        self.assertEqual(symbols[10],"Ċ")

    def test_offline_merge_conversion_preserves_rank_order(self):
        encoding=SimpleNamespace(_mergeable_ranks={b"a":0,b"b":1,b"ab":2},_pat_str="test",_special_tokens={"<eos>":3})
        converted=prepared_bpe(encoding)
        self.assertEqual(converted["vocab"],{"a":0,"b":1,"ab":2})
        self.assertEqual(converted["merges"],[["a","b"]])
        self.assertEqual(len(converted["ranks_sha256"]),64)


if __name__=="__main__":
    unittest.main()
