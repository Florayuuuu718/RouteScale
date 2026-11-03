"""CPU tensor checks for the TP LM head ordering used before Graph capture."""
import unittest
import torch
from serving.tp_precision import gather_head


class GatherTests(unittest.TestCase):
    def test_batch_two_rank_major_gather_becomes_batch_major_and_trims(self):
        class Comm:
            def all_gather(self,x):
                return torch.tensor([[0,1,2],[10,11,12],[3,4,5],[13,14,15]])
        class Head:tp_size=2;num_embeddings=5;_comm=Comm()
        out=gather_head(torch.zeros(2,3),Head())
        self.assertTrue(torch.equal(out,torch.tensor([[0,1,2,3,4],[10,11,12,13,14]])))

    def test_single_batch_and_four_ranks_keep_all_vocab_shards(self):
        class Comm:
            def all_gather(self,x):return torch.arange(12).reshape(4,3)
        class Head:tp_size=4;num_embeddings=11;_comm=Comm()
        self.assertTrue(torch.equal(gather_head(torch.zeros(1,3),Head()),torch.arange(11).reshape(1,11)))

    def test_tp1_returns_tensor_without_a_collective(self):
        class Head:tp_size=1
        x=torch.randn(3,7);self.assertIs(gather_head(x,Head()),x)


if __name__=='__main__':unittest.main()
