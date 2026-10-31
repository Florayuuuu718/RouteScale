# H 阶段 nano-vLLM 实验协议

## H1 固定对象

配置以 [`config/serving/h1_nanovllm.json`](../config/serving/h1_nanovllm.json) 为准。

- nano-vLLM：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`，MIT；
- Qwen/Qwen3-0.6B：`c1899de289a04d12100db370d81485cdf75e47ca`，Apache-2.0；
- Python 3.11.16，PyTorch 2.8.0+cu129，Transformers 4.57.6；
- FlashAttention 2.8.3.post2，使用官方 cp311 / torch2.8 / CXX11 ABI wheel，SHA256
  固定为 `92aabf18ecfcf5d620ac7fe97fafe88f83e714b7a790e4bf219dff4a404cfb38`；
- 单卡 TP=1，`enforce_eager=True`，1024-token window，最大 8 个序列，GPU memory
  utilization 0.5，block size 256；
- 上游 example.py 中的两个 user prompt，temperature=0.6，最多 64 个输出 token，
  固定 seed=20261023；使用同一个固定 tokenizer，关闭 thinking 以便快速检查输出。

上游 `enforce_eager` 只控制 ModelRunner 的 CUDA Graph 路径；Sampler、RMSNorm 等层
仍有 `torch.compile` 装饰器，因此不能将这一配置称为“完全不编译”。

## 隔离环境与目录

| 内容 | 位置 |
|---|---|
| 上游源码 checkout | `runs/vendor/nano-vllm/` |
| Python 环境 | `runs/envs/nano-vllm/` |
| 模型快照 | `runs/models/Qwen3-0.6B/` |
| 下载的 wheel | `runs/downloads/` |
| 运行日志与编译缓存 | `runs/logs/h_nanovllm/`、`runs/cache/h_nanovllm/` |
| 可审查的结果 JSON/报告 | `results/h_nanovllm/` |

训练 `.venv` 不安装上游依赖；上游源码保持 clean checkout。入口校验 commit，保存
源码、模型文件、tokenizer 和实际 workload 哈希。模型权重额外核对 HF LFS SHA256。

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git runs/vendor/nano-vllm
git -C runs/vendor/nano-vllm checkout --detach bb823b3e06983d71485a8e1f23715ebd87d98ef8
uv venv runs/envs/nano-vllm --python 3.11.16 --seed
uv pip install --python runs/envs/nano-vllm/bin/python torch==2.8.0 --torch-backend=cu129
uv pip install --python runs/envs/nano-vllm/bin/python \
  transformers==4.57.6 huggingface-hub==0.36.2 xxhash==4.0.2
python scripts/fetch_h1_wheel.py
uv pip install --python runs/envs/nano-vllm/bin/python --no-deps \
  'runs/downloads/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
uv pip install --python runs/envs/nano-vllm/bin/python einops==0.8.2
uv pip install --python runs/envs/nano-vllm/bin/python --no-deps -e runs/vendor/nano-vllm
runs/envs/nano-vllm/bin/hf download Qwen/Qwen3-0.6B \
  --revision=c1899de289a04d12100db370d81485cdf75e47ca --local-dir=runs/models/Qwen3-0.6B
```

## H1 验收顺序

1. `python -m serving.verify_nanovllm_state`：直接测试 pinned upstream 的真实
   BlockManager、Sequence 和 Scheduler。为避免包初始化提前导入 GPU engine，状态测试
   仅绕过 `nanovllm.__init__`，不修改被测模块。应覆盖引用计数、冷/热缓存、淘汰、
   prefix hash 链、尾 block、耗尽、decode 抢占、chunked Prefill 和 EOS/长度结束回收。
2. `python -m serving.verify_nanovllm_attention`：在真实 GPU 上比较 FP16/BF16 的
   varlen causal Prefill 与 FP32 SDPA reference；验证 Triton KV store 精确写入非零
   page，再比较使用 prefix cache 的 paged Prefill 和 paged Decode。任何失败先保留
   traceback，不替换 attention 后端。
3. `python -m serving.reproduce_nanovllm`：新进程加载 pinned 模型，完成两个请求，保存
   token/text，并验证 running/waiting 队列、已使用 blocks 清零。

以上命令均使用 `runs/envs/nano-vllm/bin/python`。初始化和首次生成 wall time
只作诊断：它们包含 JIT、模型加载和冷缓存，不进入正式吞吐排名。H1 不使用上游 README
数字作为本机结果，也不与 G3 的不同模型/框架版本直接排名。

本机准备了带校验值的分段下载后，可以执行
`.venv/bin/python -m serving.run_h1 --wait-for-downloads` 自动衔接后续步骤。
该入口不启动下载，等待已启动的下载原子落盘，核对 SHA256 后才安装或加载；
状态写入 `results/h_nanovllm/h1_run.json`，日志写入 `runs/logs/h_nanovllm/`。
只启动一个 runner，任一步失败都保留日志并停止。

## H2 后续正式消融

H1 通过后，先冻结共同请求 token 集合、采样参数、warmup、重复次数、计时边界和显存
预算，再分别比较 CUDA Graph 和 Prefix Cache。每个 variant 至少三个独立进程。任何
关闭 Prefix Cache 的改动都以可重放 patch 保存，避免改变采样或模型计算。

默认 256-token block 下，128-token prompt 的 75% 共享前缀不足一整块，属于不应命中的
负对照；512-token prompt 才能提供至少 256 个可复用 token。消融必须记录实际 cached
tokens/hit rate，而不是仅根据 workload 的共享比例标记命中。

TP=2/4 留给多卡机器；H1 的单卡复现与 block 状态测试不依赖四卡。RouteScale Dense
接入在 H3 单独进行；G 阶段的 BF16 greedy 分叉应以 logits 容差和首个差异定位处理。

H2 已完成的 CPU fixture、命中 probe、首轮计划与正式结果 schema 见
[`h2_ablation_protocol.md`](h2_ablation_protocol.md)。
GPU 入口已实现；原 Prefix Cache 和 batch=1 Graph 数值 gate 未通过并保留失败证据。
batch=8 Graph 在独立 scope/目录完成 18/18 正式进程与审计，不能将其结果解读为整个
H2、batch=1 或 Prefix Cache 的验收通过。
后续两项差异已在独立新策略下完成修正：固定页表宽度、统一 paged Prefill，完整
64 步的 2,048 个张量对照精确一致。入口与记录见
[`h2_numerical_fixes.md`](h2_numerical_fixes.md)。原 gate 保持历史状态，新策略需单独计时。
新策略的四 case、三变体、三进程对照由 `scripts/run_h2_corrected_pilot.py` 执行，
状态与结果只写入 [`h2_corrected_pilot/`](../results/h_nanovllm/h2_corrected_pilot/README.md)，
36/36 正式进程与审计均通过，不覆盖历史证据。该轮只覆盖 Graph 和 Prefix 两项消融，不代表连续批处理、分页碎片
或多卡 TP 对照已经完成。

随后 fixed-cohort / continuous-admission 的混合场景新门禁发现 Prefill GEMM 行形状
舍入差异。完整 tile512 策略在两侧统一应用后，1,664 个全步向量精确一致，18/18 正式
进程与独立验证通过。吞吐/排队收益和 ITL 尾部退化见
[`调度实验`](../results/h_nanovllm/h2_scheduling_tile512/README.md)；该策略不是原生上游
性能，也不覆盖任意到达流、HTTP/SLO、分页容量/碎片或多卡。

再后续[KV 分配容量与碎片](h2_kv_capacity_protocol.md)已完成独立检查：相同 896 MiB
池，18 个 CPU 容量点、13 个真实 GPU tensor 地址检查，以及 1,056 个跨页 logits
向量全部 exact。结果区分了预留空闲、页尾浪费和连续空洞；未测吞吐或最大服务
并发。

后续[受控 KV 配额的真实请求实验](h2_kv_pressure_protocol.md)已完成新数值门禁
4,160 个 exact 向量、18 个正式进程及审计。固定预留/分页在 448 MiB 池中分别
真实执行 4/8 请求一批，吞吐 1.950× / 1.887×；896 MiB 充裕预算负对照为 0.931×。
已知最大输出预算下先资助增长，不做未知长度流量、抢占/换出或整卡最大并发搜索。
后续[负对照诊断](h2_kv_diagnostic_protocol.md)已完成 48 条同进程平衡配对 trace、
27 条受控页地址 stream 探针及 3072 个 exact 向量。旧 0.931× 退化未稳定复现，
新配对中位数为 1.017×；没有将旧结果改为收益，也未确认唯一历史根因或修改分配器。
多卡 TP 与 miniSGLang 仍是后续工作；随后 H3 首轮接入按独立合同完成，见下节。

## H3 首轮本项目 Dense 接入（2026-10-24）

已使用 G1 原 artifact，保留 learned absolute positions、LayerNorm/GELU 与权重共享，
以显式 TP=1 bridge 接入上游调度、分页 attention、Sampler 和 Decode Graph，不改
Qwen 入口。三种变体共 792 个原生 BF16 全词表向量通过，264 个 eager/Graph 向量
bitwise exact；新进程独立重算参考，权重/API/边界/关闭后资源回收也通过。

详细范围与命令见 [`h3_routescale_protocol.md`](h3_routescale_protocol.md)，结果见
[`h3_routescale/README.md`](../results/h_nanovllm/h3_routescale/README.md)。本轮仅验收
接入正确性：同环境、同 dtype 的性能合同尚未冻结，不能与 G3/Qwen 历史速度直接
排名，也不覆盖 MoE、多卡、超窗口重建或整个 H3。
