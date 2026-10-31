"""Explicit TP=1 RouteScale bridge; never patches the Qwen/upstream constructor."""

from __future__ import annotations

import atexit
from types import SimpleNamespace

from serving.artifact import load_inference_artifact
from serving.h3_model import PackedRouteScaleGPT,validate_request


def create_engine(config,artifact_path,tokenizer,*,enforce_eager=True,prefix_cache=False):
    import torch
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.layers.attention import Attention
    from nanovllm.layers.sampler import Sampler
    from nanovllm.utils.context import get_context
    from serving.h2_numerical_policy import install_shape_policy
    from serving.h2_prefix_probe import prefix_disabled_manager
    from nanovllm.engine.block_manager import BlockManager

    if config["tensor_parallel_size"]!=1 or config["max_model_len"]!=512 or config["kvcache_block_size"]!=256:
        raise ValueError("This bridge qualifies only TP=1 and the original 512-token learned-position window")
    native,metadata=load_inference_artifact(artifact_path,device="cpu",dtype=torch.float32)
    native=native.to(device="cuda",dtype=torch.bfloat16).eval()
    if native.config.block_size!=512 or native.config.vocab_size!=tokenizer.vocab_size:
        raise ValueError("Tokenizer or position capacity differs from trained Dense GPT")
    hf=SimpleNamespace(hidden_size=native.config.n_embd,num_hidden_layers=native.config.n_layer,
        num_key_value_heads=native.config.n_head,head_dim=native.config.n_embd//native.config.n_head,
        dtype=torch.bfloat16,max_position_embeddings=native.config.block_size)
    cfg=SimpleNamespace(**config,hf_config=hf,eos=tokenizer.eos_token_id,enforce_eager=enforce_eager)
    vocabulary_size=native.config.vocab_size
    native_holder=[native]

    class DenseRunner(ModelRunner):
        def __init__(self):
            # ModelRunner's stock constructor hardcodes Qwen3. Retain its prepare,
            # schedule-facing run, sampler and Graph methods; instantiate THIS
            # model directly, without AutoConfig, HF weight conversion or NCCL.
            self.config,self.block_size=cfg,256
            self.rank,self.world_size,self.event=0,1,[]
            self.enforce_eager=enforce_eager
            self.model=PackedRouteScaleGPT(native_holder.pop(),Attention,get_context).eval()
            self.sampler=Sampler()
            self.warmup_model()
            self.allocate_kv_cache()
            if not enforce_eager:
                old_device,old_dtype=torch.get_default_device(),torch.get_default_dtype()
                try:
                    torch.set_default_device("cuda")
                    torch.set_default_dtype(torch.bfloat16)
                    self.capture_cudagraph()
                finally:
                    torch.set_default_device(old_device)
                    torch.set_default_dtype(old_dtype)
            self.restore_shape=install_shape_policy(self,static_tables=True,paged_prefill=True)

        def allocate_kv_cache(self):
            shape=(2,hf.num_hidden_layers,cfg.num_kvcache_blocks,256,hf.num_key_value_heads,hf.head_dim)
            required=2*hf.num_hidden_layers*cfg.num_kvcache_blocks*256*hf.num_key_value_heads*hf.head_dim*2
            if torch.cuda.mem_get_info()[0]<required+536870912:
                raise RuntimeError("Insufficient GPU headroom for Dense KV tensor")
            self.kv_cache=torch.empty(shape,device="cuda",dtype=torch.bfloat16)
            for index,attention in enumerate(self.model.attentions):
                attention.k_cache,attention.v_cache=self.kv_cache[0,index],self.kv_cache[1,index]

        def exit(self):
            self.restore_shape()
            del self.restore_shape
            if not self.enforce_eager:
                del self.graphs,self.graph_pool
            torch.cuda.synchronize()

    class DenseEngine(LLMEngine):
        def __init__(self):
            self.ps,self.events=[],[]
            self.tokenizer=tokenizer
            Sequence.block_size=256
            self.model_runner=DenseRunner()
            self.scheduler=Scheduler(cfg)
            self._h2_num_blocks=cfg.num_kvcache_blocks
            self._h3_prefix_cache=prefix_cache
            self._h3_metadata=metadata
            self.reset_cache()
            atexit.register(self.exit)

        def reset_cache(self):
            from serving.h2_runtime import assert_released
            assert_released(self)
            cls=BlockManager if prefix_cache else prefix_disabled_manager(BlockManager)
            self.scheduler.block_manager=cls(cfg.num_kvcache_blocks,256)

        def add_request(self,prompt,sampling_params):
            tokens=self.tokenizer.encode(prompt) if isinstance(prompt,str) else list(prompt)
            validate_request(tokens,sampling_params,window=512,vocab_size=vocabulary_size)
            return super().add_request(tokens,sampling_params)

    return DenseEngine()


def close_engine(engine):
    atexit.unregister(engine.exit)
    engine.exit()
