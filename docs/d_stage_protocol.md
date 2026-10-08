# RouteScale D 阶段实验协议

## D0：冻结条件

D 阶段研究单卡 Top-1 Mixture-of-Experts。Expert Parallel 和 All-to-All
属于 F 阶段，不进入本阶段。

正式 Dense/MoE 对照固定以下条件：

- 数据为 `tinystories_full`，序列长度 512，BF16；
- 目标设备为单张 NVIDIA GeForce RTX 5060 Laptop GPU；
- 每次 optimizer update 处理 16,384 个 token；
- 首轮算法对照统一使用 `compile=False`，编译兼容性作为独立变量；
- 只替换 8 层模型中索引为 4 的一个 Transformer Block MLP；
- 复用 B1 的固定数据 RNG、窗口哈希、20 次预热、100 次测量和三次独立进程协议；
- Dense/MoE 使用同一随机种子。模型先完成 Dense 初始化，再替换目标 MLP，
  从而保持公共参数完全相同；expert 0 复制被替换 MLP 的参数；
- 如 MoE 在 batch size 8 下 OOM，Dense/MoE 同时降低 micro-batch，并提高
  gradient accumulation，保持每次 update 的 token 数不变；
- 正式结果同时报告总参数量、每 token 激活参数量、step time、tokens/s、
  allocated/reserved 显存、LM loss 和路由负载。

## D1：最小实现边界

D1 只实现本地 Top-1 路由：router 对每个 token 产生 expert 概率，选择一个
expert，按 expert 收集 token，计算后恢复原顺序。选中概率乘到 expert 输出上，
使任务 loss 能向 router 传播梯度。

D1 的最小边界不启用 capacity、token drop 或 balance loss。默认
`moe_num_experts=0` 保持原 Dense 模型；D1 smoke 配置为
`config/train_moe_tinystories_debug.py`。这些机制在 D2 中以独立开关加入。

验收条件：

1. 每个 token 恰好路由一次，输出顺序正确；
2. `num_experts=1` 与相同权重的 Dense MLP 输出和梯度一致；
3. 多 expert 时 router 和所有被选 expert 获得有限、非零梯度；
4. 未被选择的 expert 不获得梯度；
5. GPT 接入后保持 `(logits, loss)` 接口，loss 有限；
6. MoE 结构参数进入 checkpoint，旧 Dense checkpoint 仍按 Dense 模型恢复。

## D2：路由负载与机制隔离

D2 使用三个互相独立的配置项：

- `moe_capacity_factor=0.0` 表示无界路由；正数按
  `ceil(factor * tokens / experts)` 计算每个 expert 的 capacity；
- capacity 本身只记录 overflow，不改变 token 计算；只有
  `moe_drop_tokens=True` 才按展平后的稳定 token 顺序保留 capacity 内 token，
  超额 token 的 MoE 分支输出为零，由 residual 路径保留输入；
- `moe_balance_loss_weight` 控制 Switch-style balance loss 权重。Balance loss
  使用 drop 前的完整路由分布计算，硬路由比例停止梯度，router 概率负责传递梯度。

每个 update 记录 expert counts、processed counts、overflow counts、capacity、
drop rate、max/mean、负载变异系数、归一化负载熵、router 平均概率，以及分离的
LM loss、balance loss 和 total loss。正式吞吐测量必须关闭逐 update 指标落盘，
避免 GPU 同步和文件 I/O 污染计时。

D2 机制实验按以下顺序每次只增加一个变量：

1. 无界路由；
2. `capacity_factor=1.25`，只观测 overflow；
3. 保持相同 capacity，启用 token drop；
4. 保持 capacity 和 drop，加入权重为 `0.01` 的 balance loss。

四组实验使用 `config/train_moe_d2_debug.py`、相同模型初始化和相同窗口哈希。
原始数据及结论见 `results/d2_moe/`。这是机制正确性和短程行为实验，不替代
D3 的 TinyStories full 三次独立正式对照。

## D3：正式 Dense/MoE 对照与稳定性

D3 已按 D0 冻结协议完成。正式计时关闭逐 update 路由落盘，Dense 与最终
MoE 各运行三个独立进程，每个进程预热 20 updates、测量 100 updates。两边
使用相同 TinyStories full 窗口、batch、梯度累积、BF16、`compile=False`、
optimizer 和目标 GPU。

最终 MoE 配置为一个 MoE block、四个 experts、capacity factor 1.25、启用
token drop、balance-loss weight 0.01。Dense 总/激活参数均为 51,168,256；
MoE 总参数为 57,461,760，每 token 激活参数为 51,170,304。

三次运行中位数的中位数：

| Variant | Step time | Tokens/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Dense | 347.190 ms | 47,190 | 3,907.0 MiB | 4,912.0 MiB |
| Top-1 MoE | 391.047 ms | 41,898 | 4,032.8 MiB | 4,962.0 MiB |

MoE 吞吐为 Dense 的 88.78%，step time 增加 12.63%。稀疏激活没有自动转化为
更高单卡吞吐，因为 Python expert 循环、动态索引、token gather/scatter 和
router 都产生额外开销；D 阶段的价值是以接近的激活参数量增加总容量，而不是
证明当前朴素实现更快。

稳定性对照各运行 300 updates，使用相同 9,600 个训练窗口。Dense validation
LM loss 从 10.9115 降至 3.6436，MoE 从 10.9295 降至 3.6250。MoE 全程聚合
load CV 为 0.0361、drop rate 为 1.88%；最后 100 updates 的 drop rate 为 0。
单次 300-update 结果不能证明 MoE 质量更优，只证明当前配置无 NaN/OOM、负载
没有长期塌缩，且能达到与 Dense 接近的验证 loss。

原始结果、命令、限制和机器汇总见 `results/d3_moe/`。D3 的
eager 算法对照验收完成，编译兼容性作为 D4 的独立变量。

## D4：`torch.compile` 兼容性与路由边界

D4 只把 `compile=False` 改为 `compile=True`，复用 D3 eager 基线，并对
Dense 和最终 MoE 各运行三个独立 compiled 进程。数据窗口、模型、
BF16、batch、梯度累积、预热/测量次数和源码哈希已校验一致。

| Variant | Eager | Compiled | Compiled throughput change |
|---|---:|---:|---:|
| Dense | 347.190 ms / 47,190 token/s | 240.432 ms / 68,144 token/s | +44.40% |
| Top-1 MoE | 391.047 ms / 41,898 token/s | 395.580 ms / 41,418 token/s | -1.15% |

Dense 获得 1.444 倍加速，而 MoE 只有 0.989 倍。Profiler 显示 compiled MoE 的
copy 调用显著减少，但 CUDA self time 只降低 0.7%，CPU self time 增加
5.0%；`nonzero`、`bincount`、`index_select/index_copy` 仍保留。
`TORCH_LOGS=graph_breaks,recompiles` 确认 `moe.py:153` 的 `bincount`、
`moe.py:176` 的数据依赖 `nonzero` 和 expert 循环形成 graph break，动态
expert batch 尺寸还会触发 shape recompilation。

因此当前 MoE 不默认开启 `torch.compile`。后续若优化，应先重构为可
编译的 grouped/padded dispatch，而不是直接为现有 Python 循环写 Triton 内核。
原始计时、Profiler 汇总、graph-break 记录和完整限制见
`results/d4_moe_compile/`。D 阶段验收至此完成。
