import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from serving.i2_attention_reference import causal_math,metadata_checks,tensor_metrics
from serving.i2_numerical_policy import install_prefill_no_split
from serving.i2_linear_tile_policy import tiled_linear


class AttentionReferenceTests(unittest.TestCase):
    def test_suffix_causal_mask_uses_absolute_query_positions(self):
        q=torch.zeros(2,2,1);k=torch.zeros(5,1,1);v=torch.arange(5.).view(5,1,1)
        actual=causal_math(q,k,v,query_start=3,dtype=torch.float64)
        expected=torch.tensor([1.5,2.],dtype=torch.float64)[:,None,None].expand(2,2,1)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
        full=causal_math(torch.zeros(5,2,1),k,v,query_start=0,dtype=torch.float64)
        self.assertTrue(torch.equal(actual,full[-2:]))

    def test_attention_rejects_misaligned_positions(self):
        with self.assertRaises(ValueError):causal_math(torch.ones(2,2,1),torch.ones(5,1,1),torch.ones(5,1,1),4)

    def test_metadata_rejects_duplicate_pages_wrong_positions_or_writes(self):
        meta=dict(cached=3,extend=2,q_indptr=[0,2],kv_indptr=[0,5],seq_lens=[5],
            positions=torch.tensor([3,4]),input_ids=torch.tensor([13,14]),
            indices=torch.tensor([4,2,5,0,1]),out_loc=torch.tensor([0,1]))
        self.assertTrue(all(metadata_checks(meta,list(range(10,15)),3,6).values()))
        meta['out_loc']=torch.tensor([1,0])
        self.assertFalse(metadata_checks(meta,list(range(10,15)),3,6)['write_locations'])
        meta['indices'][1]=4
        self.assertFalse(metadata_checks(meta,list(range(10,15)),3,6)['indices_unique'])
        meta['positions']=torch.tensor([0,1])
        self.assertFalse(metadata_checks(meta,list(range(10,15)),3,6)['positions'])

    def test_full_tensor_comparison_catches_one_value_and_nonfinite(self):
        a=torch.ones(4,3);b=a.clone();b[-1,-1]=2
        result=tensor_metrics(a,b)
        self.assertFalse(result['exact']);self.assertEqual(result['different_elements'],1)
        b[-1,-1]=float('nan');self.assertFalse(tensor_metrics(a,b)['finite'])

    def test_policy_modifies_prefill_only_and_can_be_restored(self):
        calls=[]
        native=lambda **kwargs:calls.append(kwargs)
        decode=lambda **kwargs:None
        backend=type('FlashInferBackend',(),{})()
        backend.prefill_wrapper=SimpleNamespace(plan=native)
        backend.decode_wrappers=SimpleNamespace(plan=decode)
        restore=install_prefill_no_split(SimpleNamespace(attn_backend=backend))
        backend.prefill_wrapper.plan(causal=True,disable_split_kv=False)
        self.assertEqual(calls,[{'causal':True,'disable_split_kv':True}])
        self.assertIs(backend.decode_wrappers.plan,decode)
        with self.assertRaises(ValueError):backend.prefill_wrapper.plan(fixed_split_size=128)
        restore();self.assertIs(backend.prefill_wrapper.plan,native)

    def test_policy_rejects_another_backend(self):
        with self.assertRaises(ValueError):install_prefill_no_split(SimpleNamespace(attn_backend=object()))

    def test_linear_tiles_preserve_rows_bias_and_order(self):
        x=torch.arange(21,dtype=torch.float64).view(7,3)
        weight=torch.arange(12,dtype=torch.float64).view(4,3)
        bias=torch.arange(4,dtype=torch.float64);calls=[]
        def linear(value):
            calls.append(value.clone())
            return torch.nn.functional.linear(value,weight,bias)
        actual=tiled_linear(x,linear,tile=5)
        expected=torch.nn.functional.linear(x,weight,bias)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual([len(c) for c in calls],[5,5])
        self.assertTrue(torch.equal(calls[1][:2],x[5:]))
        self.assertFalse(bool(calls[1][2:].any()))

    def test_linear_tiles_reject_empty_or_invalid_geometry(self):
        for x,tile in ((torch.empty(0,3),5),(torch.ones(2,3),0),(torch.ones(2,3,4),5)):
            with self.assertRaises(ValueError):tiled_linear(x,lambda value:value,tile)


if __name__=='__main__':unittest.main()
