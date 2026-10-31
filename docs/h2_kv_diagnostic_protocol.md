# H2：充裕预算负结果的配对与页地址诊断

配置：[h2_kv_diagnostic.json](../config/serving/h2_kv_diagnostic.json)。
原始结果：[h2_kv_pressure/](../results/h_nanovllm/h2_kv_pressure/)。
本轮只诊断 `ample_short` 的 0.931× 观察，不修运行时、不重写原 18-process 数据，
也不修改教学大纲。数值、完整请求吞吐、CUDA stream 探针分别标记证据类型。

## 旧记录能说明什么

原始配额实验的 batch、模型调用结构、相同 seed 的采样结果均一致，编译计数不变，
但分页侧部分 Prefill/Decode 的主机可见调用时间更长。该区间包含输入准备、模型、
采样、copy 与 postprocess；它不能直接指出 attention 内核慢、页局部性差或发生
GPU 降频。新实验前会核对旧 gate/audit 与全部原始文件的 SHA256。

## 第一层：完整请求的同进程配对

每个独立进程只初始化一次模型与同一个 896 MiB/32 页 tensor，保留已合格的
BF16、Prefill tile512、Decode Graph、FCFS fixed-cohort、16×128 输入与 64 输出。
先给两侧各完整预热一次，再执行四个交替区组：

```text
区组 0：static → paged → paged → static
区组 1：paged → static → static → paged
```

后两组重复此平衡方式；第二个独立进程反转初始顺序。两策略各占两次，平均顺序
位置相同，同时轮换内外位置，减少单向运行次序与单次初始化状态的混淆。
每个区组四次都重置为同一个 seed；沿用原完整请求执行与校验函数，没有替换
allocation、attention、sampler 或计时边界。要求同区组四次的逐请求采样哈希一致。

共三个独立进程、12 个区组、48 条完整请求 trace。每区组以两侧相同输出总量除以
各自两条 trace 的 wall time 算配对吞吐比，再对四个区组/三个进程分别取中位数。
保留所有区组值，不用复测数字替换原三次独立运行的 0.931×。

## 第二层：单独控制当前 KV 的物理页号

这里不再比较两种预留策略，统一用一个 prefix-disabled 上游 BlockManager，只调整
初始空闲页顺序。8 个请求全部经过真实 allocate/schedule/postprocess/deallocate，
每请求当前只用一页；私有的未来预留不再是变量。

| 布局 | 八请求的当前页号 | 对应原实验的主要模式 |
|---|---|---|
| stride4 | 0,4,8,12,16,20,24,28 | 固定四页窗口的当前页 |
| dense0 | 0,1,2,3,4,5,6,7 | 上游按需分配的首批 |
| dense8 | 8,9,10,11,12,13,14,15 | 上游按需分配的第二批 |

固定原第一批八个 prompt、全部 64 个输出位置，用同一 reversed-prompt teacher
forcing。每进程先保存三种布局的完整 BF16 logits，cache 填 NaN，验证有效 KV
全部有限、未拥有页仍为 NaN，所有引用释放；dense0/dense8 相对 stride4 的
1024 个全词表向量必须 exact，才允许该进程的布局计时。

每种布局每进程测三次，使用三阶循环顺序使布局各占一次首/中/末位置，共 27 个
探针。CUDA event 包住 `ModelRunner.run_model`：包含模型、Graph buffer copy 与
LM head，排除输入准备和 sampler。每步同步后记录 event elapsed 与主机耗时。
stream 时间也可能包含主机 dispatch 导致的 GPU 空档，尤其是 eager Prefill，
不能叫作 attention-kernel-only 时间。

这层不执行 sampler，不能把其输出 token 数除以耗时冒充服务 tokens/s。独立审计
重新读取保存张量，对三个进程合计 3072 个全词表向量作完整比较。

## 运行与约束

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m unittest tests.test_h2_kv_diagnostic
runs/envs/nano-vllm/bin/python scripts/run_h2_kv_diagnostic.py
runs/envs/nano-vllm/bin/python -m serving.h2_kv_diagnostic --check-only
```

- 可审查 JSON：`results/h_nanovllm/h2_kv_diagnostic/`。
- 大型 logits：`runs/validation/h2_kv_diagnostic/h2_kv_diagnostic/`。
- 日志：`runs/logs/h_nanovllm/h2_kv_diagnostic/`。

使用公共 GPU 锁串行执行；结果/张量不覆盖，失败保留；计时内不许新增编译或
静默 fallback。GPU 频率、功率、P-state 与温度只在区组之间读取，不改电源设置、
不锁频，因此不能排除区组内部的瞬时设备/主机状态变化。

若配对与地址探针都不稳定重现 6.9% 退化，应报告“不支持稳定布局惩罚”，而非
伪造某个唯一根因；若稳定重现，再缩小到具体算子或分配器。原负结果始终保留。
