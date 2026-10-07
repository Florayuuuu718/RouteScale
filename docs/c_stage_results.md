# C 阶段四卡实验结果与训练路径

## 结论

C 阶段已经完成。环境准入、DDP 数值正确性、1/2/4 卡 strong/weak
scaling、通信 trace、完整数据覆盖与恢复、DeepSpeed ZeRO-0/1/2/3、
FSDP2、分片 checkpoint 和容量边界都有机器实测证据。

没有一个后端在所有目标上都最好：

- **50.9M 小模型单卡**：原生 PyTorch 最快，189,198 token/s；没有理由为分片支付额外开销。
- **50.9M 小模型四卡吞吐**：ZeRO-0 最快，357,031 token/s；是原生 DDP 的 1.114 倍。ZeRO-1 次之，334,070 token/s，并且四卡峰值 allocated 最低。
- **固定全局 batch 的原生训练**：DDP 路径最透明，已完成完整一遍数据覆盖与精确恢复；四卡 strong speedup 为 1.694 倍。
- **模型放不进 DDP 时**：先尝试 ZeRO-2；本机测试成功到 20.41 亿参数。超过该点再用 ZeRO-3；它成功到 22.74 亿参数，但小模型吞吐明显更低。
- **希望保持 PyTorch 原生分片栈时**：使用 FSDP2。它在 DDP 已 OOM 的 9.84 亿参数点成功；同拓扑下比 ZeRO-3 少用 GPU 显存，但吞吐更低且 CPU RSS 更高。本实验没有继续搜索它的最终 OOM 边界。

这些结论是“给定硬件、模型和协议下的路径选择”，不是所有模型、网络和版本上的普遍排名。

## 实验边界和统一口径

正式性能矩阵使用 4 张 NVIDIA GeForce RTX 4090 D，每张 24,564 MiB，
PyTorch 2.12.1+cu130、CUDA Toolkit 13.0、NCCL 2.29.7 和 DeepSpeed
0.19.7。服务器没有 NVLink，GPU 通过 PCIe/双 NUMA 主机互联。

50.9M 基准的精确总参数量为 51,168,256，配置为 8 层、8 头、宽度
512、序列长度 512、BF16、每卡 micro-batch 8。正式吞吐采用 20 次预热、
100 次测量、3 个独立进程，报告三次“最慢 rank step 中位数”的中位数。
`torch.compile`、Profiler、validation 和 checkpoint I/O 均不在正式计时区间。

实例重开后有一张物理 GPU 被替换，拓扑由“NUMA 0 两张 + NUMA 1 两张”
变为“NUMA 0 一张 + NUMA 1 三张”。正式 DDP、后端吞吐矩阵和 DDP trace
来自第一次拓扑；DeepSpeed/FSDP2 后端 trace、FSDP2 容量点、DDP RSS 补测
和 ZeRO-3 失败点来自重开后的拓扑。因此不能把两次会话的绝对 step 时间混成同一性能排名，
容量是否成功和显存边界仍可作为对应配置的实测证据。

## 从问题到证据

| 问题 | 控制条件 | 验收证据 | 结论 |
|---|---|---|---|
| 四卡环境是否可用 | 4 卡、NCCL、数据哈希、BF16、DeepSpeed、FSDP2 | C0 两次 preflight 必需检查全部通过 | 环境可进入正式实验；唯一警告是结果文件使 worktree 非 clean |
| rank 分片是否正确 | 同初始化、同一全局 batch | 2/4 卡更新对照 | loss 完全一致，参数最大绝对误差 `3.725e-09`，副本误差 0 |
| DDP 是否扩展 | 相同模型和计时协议 | strong/weak 各 1/2/4 卡、每点 3 次 | strong 四卡 1.694 倍；weak 四卡 3.014 倍 |
| 多卡是否真的训练完整数据 | 确定性窗口、pad 尾部、完整 validation | C5 full | 925,766 个唯一窗口覆盖率 100%，恢复检查通过 |
| 后端语义是否一致 | 相同初始化、窗口与 FP32 smoke | native/ZeRO-0/FSDP2 trajectory | 四步 loss 完全一致；ZeRO-0 参数误差 `7.451e-09` |
| 哪个小模型后端快 | 51.17M、BF16、相同全局工作量 | 五后端正式矩阵 | 四卡 ZeRO-0 最快；单卡原生最快 |
| 分片是否扩大容量 | 固定 batch、block 和四卡 | DDP/ZeRO-2/ZeRO-3/FSDP2 容量探测 | DDP 6.94 亿成功；ZeRO-2 20.41 亿；ZeRO-3 22.74 亿 |
| 分片状态能否恢复和导出 | 四卡保存、新进程加载、普通模型 strict load | checkpoint 定量矩阵 | ZeRO-2/3 与 FSDP2 均恢复；ZeRO-2/3 可合并普通 FP32 权重 |

## DDP 正确性与扩展

2 卡和 4 卡 NCCL 更新检查都通过。聚合 loss 与单进程等价全局 batch 完全
一致，更新后参数相对参考的最大绝对误差均为 `3.725e-09`，所有 rank 间
参数最大差为 0。这先证明“结果是同一个训练更新”，再允许解释速度。

### Strong scaling

每次全局更新固定 16,384 token：

| GPU | ms/update | 全机 token/s | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 86.597 | 189,198 | 1.000x | 100.0% |
| 2 | 66.884 | 244,961 | 1.295x | 64.7% |
| 4 | 51.125 | 320,468 | 1.694x | 42.3% |

四卡没有接近 4 倍并不表示 DDP 错误。全局工作固定后，每卡计算量降到四分
之一，但梯度 AllReduce、启动和同步仍然存在；没有 NVLink 且跨 NUMA 的
PCIe 拓扑进一步放大了通信占比。三次运行的最大相对 spread 仅 0.23%，
loss 全部有限且运行时窗口哈希与计划一致，因此这是稳定的系统现象。

### Weak scaling

每卡保持 16,384 token/update，全局工作随 GPU 数增加：

| GPU | ms/update | 全机 token/s | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 86.593 | 189,208 | 1.000x | 100.0% |
| 2 | 109.499 | 299,253 | 1.582x | 79.1% |
| 4 | 114.931 | 570,219 | 3.014x | 75.3% |

weak scaling 四卡吞吐增长更好，因为每卡仍有足够计算来摊薄通信；它的全局
batch 同时扩大，不能把 3.014 倍直接解释为相同训练目标更快收敛。

## 完整覆盖和恢复

C5 在四卡上对完整 TinyStories token 文件执行一次确定性覆盖：

- 925,766 个完整训练窗口全部至少使用一次，覆盖率 100%；
- 为补齐最后一个全局更新明确重复 26 个窗口，43 个训练尾部 token 未伪装成完整窗口；
- 共 28,931 次 optimizer update，耗时 3,952.23 秒；
- 完整 validation 覆盖 4,765,917 个 target，loss 从 10.9090 降至 1.4831；
- 峰值 allocated/reserved 为 3.758/4.429 GiB；
- `2 步 + 保存 + 新进程恢复到 4 步` 与连续 4 步的模型 SHA256、loss 历史、下一更新编号和窗口前缀哈希全部相同。

它证明的是数据覆盖、训练有效性和恢复语义。包含 validation 与周期 checkpoint
的 119,934 token/s 不是正式稳态吞吐，不应与前面的无 checkpoint benchmark
直接比较。

## DeepSpeed/FSDP2 正式矩阵

下表是 51.17M 模型的四卡正式结果。显存为三次运行中每次最忙 rank 峰值的
中位数：

| 后端 | ms/update | token/s | 相对原生 DDP | 4 卡 scaling efficiency | allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|
| 原生 DDP | 51.125 | 320,468 | 1.000x | 42.3% | 3.757 / 4.188 |
| ZeRO-0 | **45.890** | **357,031** | **1.114x** | 56.4% | 2.729 / 3.160 |
| ZeRO-1 | 49.044 | 334,070 | 1.042x | 57.6% | **2.298** / 3.148 |
| ZeRO-2 | 56.594 | 289,498 | 0.903x | 64.8% | 3.073 / 4.172 |
| ZeRO-3 | 107.494 | 152,417 | 0.476x | 80.8% | 3.277 / 3.787 |
| FSDP2 | 69.117 | 237,046 | 0.740x | 68.8% | 3.091 / 3.969 |

ZeRO-3 的 80.8% 是“相对它自己很慢的单卡基线”的扩展效率，不表示它是
最快后端。小模型上，参数反复 materialize 和更多集合通信没有足够计算来
摊薄，所以 ZeRO-3 比 ZeRO-0 慢 2.34 倍。ZeRO-0/1 在这个固定点更适合追求
吞吐；ZeRO-2/3 和 FSDP2 的价值主要在更大模型的状态分片，而不是保证小模型
更快。

所有正式后端运行的 loss 都有限，窗口哈希都与计划一致。显存数字包含各框架
不同的 bucket、缓存和临时 buffer，不能只按理论的“参数/梯度/optimizer 分片
比例”机械推算。

## 通信与重叠

四卡 trace 实际观察到的 NCCL kernel 为：

| 后端 | trace 中观察到的通信 | NCCL 与非 NCCL GPU kernel 直接重叠中位数 |
|---|---|---:|
| 原生 DDP | AllReduce | 22.4% |
| ZeRO-0 | AllReduce | 0.0% |
| ZeRO-1 | AllGather + AllReduce | 0.0% |
| ZeRO-2 | AllGather + AllReduce | 0.0% |
| ZeRO-3 | AllGather + ReduceScatter + 小量 AllReduce | 0.7% |
| FSDP2 | AllGather + ReduceScatter + 小量 AllReduce | 18.0% |

ZeRO-2 在本次 DeepSpeed 0.19.7 配置中实际使用了 multi-rank bucket
AllReduce，不能仅凭“ZeRO-2 概念上分片梯度”就把 trace 写成
ReduceScatter。这里以实际 kernel 为准。

DDP trace 中 60.1% 的 NCCL 时间位于标记的 backward scope 内；DeepSpeed
和 FSDP2 入口没有同名 backward `record_function` scope，因此它们的
`within_backward_scope=0` 是“不可测”，不是“通信不在反向传播”。直接重叠
指标仍显示本次 ZeRO-0/1/2/3 trace 的通信基本串行，是小模型 ZeRO-3 较慢的
证据之一。Profiler trace 只解释机制，正式速度仍由无 Profiler 矩阵裁决。

## Checkpoint 生命周期

| 后端 | 文件结构 | 总大小 | 初次保存 | 恢复加载 | 恢复后再保存 |
|---|---|---:|---:|---:|---:|
| ZeRO-2 | 1 model + 4 optimizer | 683.2 MiB | 0.434 s | 0.353 s | 0.424 s |
| ZeRO-3 | 4 model + 4 optimizer | 585.7 MiB | 0.255 s | 0.154 s | 0.294 s |
| FSDP2 DCP | 4 DCP + 4 RNG + 2 metadata | 685.0 MiB | 0.794 s | 0.711 s | 0.533 s |

ZeRO-2/3 都额外合并为约 195.2 MiB 的普通 FP32 `state_dict`，并由不依赖
DeepSpeed Engine 的 `GPT` strict load：ZeRO-2 合并 0.140 秒、峰值进程 RSS
1.365 GiB；ZeRO-3 合并 0.079 秒、峰值 RSS 1.358 GiB。这里的绝对 I/O 时间
受本地文件系统缓存影响，主要用于说明格式、分片数量和生命周期均被验证。

## 容量边界

容量探测固定四卡、BF16、micro-batch 8、block 512，采用 2 次预热和 5 次
测量。它回答“能否完成短训练更新”，不是正式吞吐排名：

| 后端 | 最大测试成功点 | 成功点显存 allocated/reserved | 成功点 CPU RSS | 首个更大失败点 | 失败位置 |
|---|---:|---:|---:|---:|---|
| 原生 DDP | 694,213,120 | 19.748 / 21.365 GiB | 3.591 GiB/rank | 984,050,688 | 首次 warmup forward OOM |
| ZeRO-2 | 2,040,626,560 | 20.887 / 22.936 GiB | 9.020 GiB/rank | 2,155,553,280 | backward bucket/collective OOM |
| ZeRO-3 | **2,273,625,728** | 22.166 / 22.939 GiB | 2.115 GiB/rank | 2,394,843,904 | `engine.backward` OOM |
| FSDP2 | 984,050,688 | 12.091 / 13.898 GiB | 4.758 GiB/rank | 未搜索 | 成功点只是下界 |

在实例重开后的同一拓扑、同一个 984,050,688 参数配置上，ZeRO-3 与
FSDP2 又做了一次直接对照：

| 后端 | ms/update | token/s | allocated/reserved GiB | CPU RSS GiB/rank |
|---|---:|---:|---:|---:|
| ZeRO-3 | **484.656** | **33,805** | 14.020 / 17.436 | **1.615** |
| FSDP2 | 728.808 | 22,481 | **12.091 / 13.898** | 4.758 |

这个点上 ZeRO-3 吞吐高 50.4%，FSDP2 则少用约 1.93 GiB allocated 和
3.54 GiB reserved GPU 显存。它说明“更省 GPU 显存”“更省 CPU 内存”和
“更快”是三个不同目标：本次 FSDP2 交换到了更低 GPU 峰值，但付出了更高
CPU RSS 和更低吞吐。

因此可以严格声称：同型号四卡上，ZeRO-2/3 让 DDP 无法启动的 9.84 亿参数
配置完成训练更新；ZeRO-3 的实测成功上界比 ZeRO-2 高约 2.33 亿参数。
不能声称 ZeRO-3 的数学极限正好是 22.74 亿，也不能把 FSDP2 的 9.84 亿
成功点写成其最大容量，因为实验只给出了离散配置的成功/失败夹逼。

## 推荐训练路径

```text
模型能放入单卡？
├─ 是：优先原生 PyTorch；需要四卡最高小模型吞吐时先测 ZeRO-0/ZeRO-1
└─ 否：模型能放入 DDP 的每卡完整副本？
   ├─ 是：优先 DDP，代码和恢复语义最直接
   └─ 否：先 ZeRO-2
      ├─ 能运行且吞吐可接受：保留 ZeRO-2
      └─ 仍 OOM：ZeRO-3 + zero.Init；接受更高通信和复杂 checkpoint 成本

需要纯 PyTorch 分片栈或 DCP？
└─ 选择 FSDP2，并按目标规模重新做容量与吞吐验证
```

实际项目还应在进入 ZeRO-3 前依次考虑减小 micro-batch、梯度累积、激活
checkpoint、Flash Attention/SDPA 和更合适的序列长度。容量后端解决的是模型
状态分布，不自动解决激活显存或训练收敛问题。

## 证据位置与限制

核心文件：

- [环境](../artifacts/c_stage_autodl/results/c0_environment/)
- [DDP 正确性](../artifacts/c_stage_autodl/results/c_ddp/correctness/)
- DDP scaling：[strong](../artifacts/c_stage_autodl/results/c_ddp_autodl/strong/summary.json) / [weak](../artifacts/c_stage_autodl/results/c_ddp_autodl/weak/summary.json)
- [完整覆盖](../artifacts/c_stage_autodl/results/c5_coverage/4gpu_full/result.json)
- [后端矩阵](../artifacts/c_stage_autodl/results/c_backend_autodl/summary.json)
- [通信算子](../artifacts/c_stage_autodl/results/c_backend_profiler/summary.json)
- [通信重叠](../artifacts/c_stage_autodl/results/c_trace_overlap/summary.json)
- [原始 Chrome traces（压缩包）](../artifacts/c_stage_autodl/routescale_c_raw_traces.tar.gz)
- [checkpoint](../artifacts/c_stage_autodl/results/c_checkpoint_metrics/summary.json)
- [容量](../artifacts/c_stage_autodl/results/c_capacity/summary.json)

证据包清单对 360 个文件逐项记录大小和 SHA256，本地复核为 360 通过、0
失败。大 checkpoint、数据集和缓存未入库；24 个原始 Chrome traces 另行
压缩保存，并在本地从原始 trace 重算出与服务器逐字一致的 overlap summary。
短 benchmark 证明运行正确和稳态
系统表现，不证明不同后端的长程收敛完全等价；只有原生 DDP 路径完成了本次
完整 TinyStories 覆盖训练。
