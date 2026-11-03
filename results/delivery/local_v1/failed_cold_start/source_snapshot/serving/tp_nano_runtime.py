"""Explicit torchrun-driven nano-vLLM TP adapter; no shared-memory worker loop.

Native model/sharding/attention/Graph code is reused. The fixed KV pool and
static page-table/paged-prefill policy are applied on EVERY rank. This is a new
internal TP agreement scope, not a claim of HF FP32 numerical equivalence.
"""
import atexit
import math
import time

import torch
import torch.distributed as dist


def create(c,variant,world,rank,numeric):
    from datetime import timedelta
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    from nanovllm.layers.sampler import Sampler
    from nanovllm.utils.loader import load_model
    from serving.h2_numerical_policy import install_shape_policy
    from serving.h2_prefix_probe import prefix_disabled_manager
    from nanovllm.engine.block_manager import BlockManager
    class Runner(ModelRunner):
        def __init__(self):
            self.config=Config(c['model_path'],tensor_parallel_size=world,enforce_eager=variant!='graph_no_prefix',
                max_model_len=1024,max_num_batched_tokens=4096,max_num_seqs=4,num_kvcache_blocks=32)
            self.config.hf_config.dtype=torch.bfloat16
            self.block_size=256;self.enforce_eager=self.config.enforce_eager;self.world_size=world;self.rank=rank
            dist.init_process_group('nccl',init_method='tcp://127.0.0.1:2333',world_size=world,rank=rank,timeout=timedelta(seconds=60))
            torch.cuda.set_device(rank);old_dtype=torch.get_default_dtype();old_device=torch.get_default_device()
            torch.set_default_dtype(torch.bfloat16);torch.set_default_device(f'cuda:{rank}')
            try:
                self.model=Qwen3ForCausalLM(self.config.hf_config);load_model(self.model,self.config.model)
                self.sampler=Sampler();self.warmup_model();self.allocate_kv_cache()
                if not self.enforce_eager:self.capture_cudagraph()
            finally:torch.set_default_device(old_device);torch.set_default_dtype(old_dtype)
            self.restore=install_shape_policy(self,static_tables=True,paged_prefill=True)
        def allocate_kv_cache(self):
            hf=self.config.hf_config
            shape=(2,hf.num_hidden_layers,32,256,hf.num_key_value_heads//world,hf.head_dim)
            self.kv_cache=torch.empty(shape,device=f'cuda:{rank}',dtype=torch.bfloat16)
            for index,module in enumerate(m for m in self.model.modules() if hasattr(m,'k_cache')):
                module.k_cache,module.v_cache=self.kv_cache[0,index],self.kv_cache[1,index]
    runner=Runner()
    class Replay:
        def __init__(self):
            self.engine=runner;self.scheduler=None
        def reset(self):
            self.scheduler=Scheduler(runner.config)
            if variant!='eager_prefix':self.scheduler.block_manager=prefix_disabled_manager(BlockManager)(32,256)
        def wave(self,requests):
            from nanovllm import SamplingParams
            from nanovllm.utils.context import reset_context
            self.scheduler=self.scheduler or Scheduler(runner.config)
            seqs=[Sequence(r['input_ids'],SamplingParams(temperature=.6,max_tokens=len(r['continuation_ids']),ignore_eos=True)) for r in requests]
            mapping={s.seq_id:r for s,r in zip(seqs,requests)};steps={s.seq_id:0 for s in seqs};values={r['request_id']:[] for r in requests}
            for s in seqs:self.scheduler.add(s)
            hits={};replays=0;torch.cuda.synchronize();started=time.monotonic()
            for _ in range(2048):
                active,prefill=self.scheduler.schedule()
                for s in active:hits.setdefault(s.seq_id,s.num_cached_tokens)
                prepared=runner.prepare_prefill(active) if prefill else runner.prepare_decode(active)
                logits=runner.run_model(*prepared,prefill)
                if not prefill and not runner.enforce_eager:replays+=1
                forced=[]
                for index,s in enumerate(active):
                    row=mapping[s.seq_id];step=steps[s.seq_id]
                    if numeric and rank==0:values[row['request_id']].append(logits[index].detach().clone())
                    forced.append(row['continuation_ids'][step]);steps[s.seq_id]+=1
                self.scheduler.postprocess(active,forced,prefill);reset_context()
                if not self.scheduler.waiting and not self.scheduler.running:break
            else:raise TimeoutError('Bounded TP replay exhausted')
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            bm=self.scheduler.block_manager
            if bm.used_block_ids or len(bm.free_block_ids)!=32:raise ValueError('Native TP KV resources unreleased')
            tensors={k:torch.stack(v).cpu().float() for k,v in values.items()} if numeric and rank==0 else {}
            return dict(seconds=elapsed,requests=[dict(request_id=r['request_id'],cached_tokens=hits[s.seq_id],
                output_tokens=len(s.completion_token_ids),actual_tokens=s.completion_token_ids) for s,r in zip(seqs,requests)],
                graph_replays=replays,released=True),tensors,[]
        def shutdown(self):
            runner.restore()
            if not runner.enforce_eager:del runner.graphs,runner.graph_pool
            torch.cuda.synchronize();dist.destroy_process_group()
    return Replay()
