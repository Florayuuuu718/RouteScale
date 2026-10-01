# RouteScale：Transformer 训练系统项目实施大纲

> 版本：2026-09-20
> 本文只记录项目路线、实施任务和阶段验收条件，不记录学习问答、完成进度或具体实验结果。

相关文档：

- 学习内容、概念解释、易错点和复习题：[`RouteScale_学习笔记.md`](RouteScale_学习笔记.md)
- 环境与硬件版本：[`environment.md`](environment.md)
- 已完成实验的配置、原始值和结论：[`../results/README.md`](../results/README.md)
- TinyStories 数据版本、规模和哈希：[`../data/tinystories_full/manifest.json`](../data/tinystories_full/manifest.json)

## 0. 项目目标与边界

### 核心问题

使用 nanoGPT 和 TinyStories 建立一条从单卡训练到多卡训练与算子优化的实验路线：

1. 单卡训练一步的时间与显存花在哪里？
2. 1/2/4 卡 DDP 的吞吐、通信和扩展效率怎样变化？
3. 将部分 Dense MLP 替换为 MoE 后，路由、负载和训练性能怎样变化？
4. 根据 trace 选择一个真实热点后，Triton 实现能否改善完整训练？

### 固定实验对象

- 训练代码基于 karpathy/nanoGPT，并记录上游 commit。
- 正式数据使用 TinyStories 已有的 train/validation split。
- Shakespeare 只用于环境冒烟检查。
- TinyStories 调试子集只用于排错，不进入正式性能对照。
- 正式 1/2/4 卡结果来自同一台四卡服务器。
- 项目重点是训练系统证据，不以最终生成质量作为主要验收目标。

### 最终交付物

- 可复现的代码、环境和数据清单；
- 一个冻结的正式模型与训练配置；
- 单卡无 Profiler 基线与 Profiler trace；
- 1/2/4 卡 strong/weak scaling 报告；
- 一次四卡完整训练窗口覆盖记录；
- 单卡 Top-1 MoE 与 Dense 对照；
- 一个 Triton 前反向正确性和完整训练性能对照；
- 项目 README、图表、结论、失败记录和简历要点。

---

## 1. 全阶段统一实验规则

### 版本与数据

1. 每次正式实验记录代码 commit、数据 revision、tokenizer 和 bin 文件 SHA256。
2. 对照实验使用相同的数据文件、窗口顺序和模型初始化。
3. 数据复制到其他机器后，先核对 SHA256，再开始实验。
4. 不向 Git 提交大型 token 文件、checkpoint、trace 或租卡凭据。

### 正确性

1. 每项性能实验开始前先做最小正确性检查。
2. 训练至少检查有限 loss、无 NaN/Inf、梯度存在和 checkpoint 可恢复。
3. 一次只修改一个主要变量。
4. 自定义训练算子必须同时验证 forward 和 backward。

### 计时

1. 正式计时前至少预热 20 次更新；运行仍不稳定时延长预热。
2. 每组正式基准至少独立重复 3 次，保留原始值并报告中位数。
3. 使用 CUDA Event 或在完整测量窗口边界同步 GPU。
4. 计时窗口不包含数据下载、validation、checkpoint、Profiler 和首次编译。
5. Profiler 运行只用于定位瓶颈，不把它的运行速度作为正式吞吐。

### 工作量

1. 每次实验核对实际的全局 tokens/update。
2. 报告全机 tokens/s、step 时间、峰值显存、loss 和实际处理 token 数。
3. 区分“数据文件包含完整数据”和“训练已经覆盖完整数据”。
4. 区分 strong scaling 与 weak scaling。
5. scaling 结果必须在同一台多卡机器上比较。

### 正式实验记录字段

| 类别 | 必须记录 |
|---|---|
| 版本 | 代码 commit、数据 revision、bin 哈希 |
| 环境 | GPU 型号和数量、驱动、PyTorch、CUDA、GPU 拓扑 |
| 模型 | 层数、头数、宽度、参数量、精度、compile 状态 |
| 工作量 | world size、micro-batch、序列长度、累积次数、tokens/update |
| 计时 | 预热步数、测量步数、重复次数、计时边界 |
| 结果 | 原始 step 时间、中位数、tokens/s、峰值显存、loss |
| 覆盖 | 实际 token 数、窗口范围、尾部与补齐规则 |
| 结论 | 变化、原因、适用条件、限制和结果路径 |

---

## A. 建立可解释的单卡基线

### A1. 固定代码和环境

#### 实施任务

1. Fork nanoGPT，保留 MIT LICENSE 和上游出处。
2. 分别设置个人仓库 `origin` 与原仓库 `upstream`。
3. 记录上游 commit、Python、PyTorch、CUDA、驱动、GPU 型号和显存。
4. 建立项目独立虚拟环境并保存依赖锁文件。
5. 使用 `compile=False` 完成 Shakespeare CUDA 冒烟检查。
6. 完成几十次训练更新，验证 forward、backward、optimizer 和 checkpoint。
7. 阅读 `model.py` 的 `CausalSelfAttention → MLP → Block → GPT`。
8. 阅读 `train.py` 的 `get_batch → forward → loss → backward → optimizer.step`。
9. 画出只包含主节点的训练流程图。

#### 验收条件

- 环境能够从依赖文件重新建立；
- CUDA 可用并确认实际 GPU；
- 训练、验证、保存和恢复链路可运行；
- 能够脱离文档解释一次完整参数更新并指出对应代码位置。

### A2. 准备 TinyStories

#### 调试数据

1. 固定 TinyStories revision、tokenizer 和故事分隔 token。
2. 使用数据集已有的 train/validation split。
3. 准备固定数量的 train/validation 调试子集。
4. 验证文本字段、tokenizer、EOT、uint16 写入和 `get_batch`。
5. 在调试配置上完成短训练和 checkpoint 检查。

#### 完整数据

1. 使用同一脚本流式处理完整 train/validation。
2. 使用有限 token 缓冲区增量写入临时文件。
3. 全部分片完成并校验后再原子改名为 `train.bin` 和 `val.bin`。
4. 生成 manifest，记录数据 revision、tokenizer、故事数、token 数、大小和 SHA256。
5. 独立复算文件大小和 SHA256。
6. 在完整文件上完成数百次单卡稳定性运行。

#### 模型候选适配

1. 选择足以形成真实 GPU 负载、又适合本地显存的候选模型。
2. 固定层数、头数、宽度、序列长度、精度和 `compile` 状态。
3. 逐步测试不同 micro-batch，记录 allocated、reserved 和 step 时间。
4. 选择保留安全显存余量的 micro-batch。
5. 使用梯度累积形成固定的 tokens/update。
6. 完成长于短跑的稳定性检查并验证 checkpoint 恢复。

#### 验收条件

- 完整 train/validation token 文件和 manifest 存在；
- 文件大小、token 数和 SHA256 可独立核对；
- 完整数据上的训练无 NaN/OOM；
- validation loss 能计算；
- checkpoint 能保存和恢复；
- 候选配置的参数量、tokens/update 和峰值显存均有记录。

---

## B. 建立可靠单卡基准并使用 PyTorch Profiler

### B1. 无 Profiler 基线

1. 实现单卡确定性窗口采样。
2. 固定窗口顺序、模型初始化、batch、序列长度、精度和 compile 状态。
3. 将预热区间、正式计时区间、评估和保存明确分开。
4. 使用 CUDA Event 或同步边界测量稳态 step。
5. 至少预热 20 次、测量 100 次更新，并独立重复 3 次。
6. 记录每次原始值、中位数、tokens/s、峰值显存、loss 和实际 token 数。
7. 在目标四卡服务器的一张固定 GPU 上重复该基准，作为 C 阶段的同机单卡基线。

### B2. Profiler trace

1. 用 `record_function` 标记 `get_batch`、forward、backward 和 optimizer step。
2. 使用短 schedule 采集 CPU/CUDA trace，例如 wait 2、warmup 2、active 10。
3. 检查 CPU 提交、GPU kernel、数据拷贝、等待和显存时间线。
4. 从 trace 中选择一个有证据的瓶颈。
5. 只修改一个与该瓶颈相关的变量。
6. 回到 B1 的无 Profiler 方法重新测量。
7. 检查训练正确性和 loss 没有异常。

### B 阶段验收

- 能打开 trace 并指出主要耗时区段；
- 能解释修改与瓶颈之间的关系；
- 有同条件、无 Profiler、至少三次重复的前后对照；
- 结果包含原始值、中位数、tokens/s、显存和正确性检查。

---

## C. 1/2/4 卡 DDP

### C1. 正确性与数据分配

1. 阅读 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、进程组和 DDP 包装代码。
2. 将确定性全局窗口序列按 rank 分配。
3. 在小文件上检查窗口是否重复、遗漏以及尾部怎样处理。
4. 核对不同 world size 下实际的全局 tokens/update。
5. 用固定输入对照单卡大 batch 与两卡分 batch 的一次参数更新。
6. 先跑通两卡，再运行四卡。

### C2. Strong scaling

1. 1/2/4 卡保持每次更新的全局工作量一致。
2. 使用相同初始化和相同全局窗口序列。
3. 使用 B1 的预热、测量和重复方式。
4. 报告全机 tokens/s、speedup、efficiency、最慢 rank step 时间和各卡峰值显存。

### C3. Weak scaling

1. 1/2/4 卡保持每卡工作量一致。
2. 使用相同模型、序列长度和精度。
3. 报告全机 tokens/s、每卡 tokens/s、通信占比和扩展效率。

### C4. 通信分析

1. 保存 NCCL/AllReduce trace。
2. 记录 `nvidia-smi -L` 和 `nvidia-smi topo -m`。
3. 区分计算量不足、通信、数据等待、负载不均和最慢 rank。
4. 检查梯度通信与 backward 是否重叠。

### C5. 四卡完整训练窗口覆盖

1. 根据窗口总数和实测吞吐估算 step、时长、存储和费用。
2. 启动前验证 checkpoint、恢复和窗口进度记录。
3. 让所有可用训练窗口至少被处理一次。
4. 记录重复补齐窗口和不能组成完整窗口的尾部 token。
5. 周期性使用固定 validation 样本检查 loss。
6. 结束时对完整 validation 文件评估一次。
7. 报告覆盖率、总时长、累计 token、峰值显存和初末 validation loss。

### C 阶段验收

- 1/2/4 卡正确性检查通过；
- strong 和 weak scaling 分开报告；
- 有 NCCL/AllReduce 与 GPU 拓扑证据；
- 能解释四卡没有达到四倍速度的原因；
- 有一次可恢复、可核对的完整训练窗口覆盖记录。

---

## D. 单卡 Top-1 MoE

### D1. 最小实现

1. 新建 `moe.py`，先替换一个 Transformer Block 中的 MLP。
2. Router 为每个 token 计算 expert 分数并选择 Top-1 expert。
3. 按 expert 收集 token、执行计算并恢复原顺序。
4. 从 2–4 个 experts 开始，不先加入 capacity 和 token drop。
5. 检查每个 token 处理一次、输出顺序正确、loss 有限。
6. 检查 router 和被选中的 expert 均有梯度。
7. `num_experts=1` 时与对应 Dense MLP 做输出和梯度对照。

### D2. 路由与负载

1. 记录每个 expert 的 token 数。
2. 记录最大/平均负载和负载离散程度。
3. 分别实验 capacity、token drop 和 balance loss。
4. 每次只增加一个机制并保留前后对照。

### D3. Dense/MoE 正式对照

1. 使用相同数据窗口、序列长度、精度、初始化和目标 GPU。
2. 同时报告总参数量和每 token 激活参数量。
3. 至少重复 3 次，比较 tokens/s、step 时间、显存和 loss。
4. 运行较长稳定性实验，观察 expert 负载和 validation loss。

### D 阶段验收

路由、输出和梯度正确；有 TinyStories 正式窗口上的 Dense/MoE 对照；能根据路由指标解释负载情况。

---

## E. Triton 热点优化

### E1. 基础准备

1. 完成 Triton vector add 和 fused softmax 教程练习。
2. 掌握 block、mask、stride、编译和预热的基本使用。
3. 回看 B/D trace，只选择一个实际热点。

### E2. 正确性

1. 从正式模型提取真实 shape 和 dtype。
2. 对照 PyTorch 检查 forward。
3. 检查输入梯度、权重梯度和 bias 梯度。
4. 覆盖正式训练会使用的 shape 和 dtype。
5. 使用最小 autograd 包装接入训练。

### E3. 性能

1. 微基准分开记录首次编译时间和预热后延迟。
2. 分别对照原生 PyTorch 与 `torch.compile`。
3. 从相同 checkpoint 和相同数据窗口开始完整训练 A/B。
4. 每组至少重复 3 次，比较 tokens/s、step 时间、显存和 loss。
5. 若整体没有收益，保留正确性与负结果并解释原因。

### E 阶段验收

forward/backward 正确；能够分别报告算子收益和完整训练收益；明确适用的 shape、dtype、硬件和限制。

---

## F. Expert Parallel，可选

进入条件：完成 D 阶段，并决定继续研究分布式 MoE。

1. 先在两卡实现 expert 分布和 All-to-All token 交换。
2. 检查路由、发送数量、输出顺序和梯度。
3. 对照单卡 MoE 与每卡复制全部 experts 的 DDP MoE。
4. 保存通信 trace、expert 负载和端到端性能。
5. 分析通信量、负载不均与模型规模的关系。

### F 阶段验收

两卡 All-to-All 正确；能够用 trace 说明通信代价；清楚区分 DDP MoE 与 Expert Parallel。

---

## G. 项目报告与求职材料

1. README 顶部说明训练任务、个人修改和核心实测结论。
2. 提供环境检查、数据准备、单卡和多卡的最短复现命令。
3. 绘制单卡时间分解、1/2/4 卡 scaling、MoE 负载和 Triton A/B 图。
4. 分开标注调试子集、完整数据短窗口和完整覆盖训练。
5. 标明 nanoGPT 上游内容、个人修改和 AI 辅助范围。
6. 保留成功、失败和无收益实验。
7. 准备 5 分钟项目讲解和简历要点。

### G 阶段验收

陌生读者能在 5 分钟内找到项目问题、方法、硬件、数据、主要图表、实测结论和结论边界。

---

## 项目文件职责

| 路径 | 职责 |
|---|---|
| `docs/RouteScale_实施教程大纲.md` | 项目路线、任务和验收条件 |
| `docs/RouteScale_学习笔记.md` | 概念、问答、易错点、判断方法和复习题 |
| `docs/environment.md` | 环境和硬件版本 |
| `results/README.md` | 已完成实验的配置、原始值和结论 |
| `data/tinystories/prepare.py` | debug/full 数据转换 |
| `data/tinystories_debug/manifest.json` | 调试数据清单 |
| `data/tinystories_full/manifest.json` | 完整数据清单 |
| `config/train_tinystories_debug.py` | 数据和训练链路排错配置 |
| `config/train_tinystories.py` | TinyStories 训练配置 |
| `model.py` | GPT 模型与后续模型接入点 |
| `train.py` | 训练、计时、Profiler 和 DDP 主入口 |
| `moe.py` | D 阶段建立的 Top-1 MoE |
