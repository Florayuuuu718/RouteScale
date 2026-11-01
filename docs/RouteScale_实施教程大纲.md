# RouteScale：Transformer 训练系统项目实施大纲

> 版本：2026-10-24
> 本文只记录项目路线、实施任务和阶段验收条件，不记录学习问答、完成进度或具体实验结果。
> 编号调整：删除原 G 总结阶段，原 H→G、I→H、J→I、K→J，子任务与过程文件同步前移。协议、脚本、配置、测试、目录和 JSON `stage` 均使用新编号，详见[迁移说明](stage_numbering_migration.md)。

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
3. DeepSpeed ZeRO 如何改变模型状态分布、显存占用、通信和吞吐，与 DDP/FSDP2 有什么差异？
4. 将部分 Dense MLP 替换为 MoE 后，路由、负载和训练性能怎样变化？
5. 根据 trace 选择一个真实热点后，Triton 实现能否改善完整训练？
6. 训练完成的 checkpoint 怎样变成带 KV Cache、连续批处理和请求调度的推理引擎？
7. nano-vLLM 与 Mini-SGLang 的推理机制在什么负载下改善吞吐、延迟和显存效率？
8. 训练阶段有效的静态布局、MoE dispatch 和并行策略，在 Prefill 与 Decode 阶段是否仍然成立？

### 固定实验对象

- 训练代码基于 karpathy/nanoGPT，并记录上游 commit。
- 正式数据使用 TinyStories 已有的 train/validation split。
- Shakespeare 只用于环境冒烟检查。
- TinyStories 调试子集只用于排错，不进入正式性能对照。
- 正式 1/2/4 卡结果来自同一台四卡服务器。
- 项目重点是训练系统证据，不以最终生成质量作为主要验收目标。
- 推理阶段先以 Dense checkpoint 建立等价性，再研究 MoE；不把训练与服务变量混在同一轮实验中。
- nano-vLLM 与 Mini-SGLang 作为固定 commit 的参考实现和对照后端，不直接复制进 RouteScale 核心源码。

### 最终交付物

- 可复现的代码、环境和数据清单；
- 一个冻结的正式模型与训练配置；
- 单卡无 Profiler 基线与 Profiler trace；
- 1/2/4 卡 strong/weak scaling 报告；
- 一次四卡完整训练窗口覆盖记录；
- DeepSpeed ZeRO-0/1/2/3 的正确性、显存、吞吐、通信和 checkpoint 对照；
- 可选的 PyTorch FSDP2 与 DeepSpeed ZeRO-3 同条件对照；
- 单卡 Top-1 MoE 与 Dense 对照；
- 一个 Triton 前反向正确性和完整训练性能对照；
- 一条从训练 checkpoint 到可验证推理产物的导出与等价性链路；
- nano-vLLM 机制实验和 Mini-SGLang 在线服务实验；
- Prefill/Decode、KV Cache、连续批处理、前缀复用、调度与 TP 的消融结果；
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

截至 2026-10-06，单卡可验证版本已由 `train_coverage.py` 完成：debug 数据的 4,222 个完整窗口覆盖率为 100%，最后一个全局更新重复补齐 2 个窗口，413 个训练尾部 token 单独记录；完整 validation 的 194,558 个 target token 全部纳入初末评估。2 步保存后恢复到 4 步与连续 4 步的模型哈希、loss 历史和窗口哈希完全一致。该结果验证机制，不替代四卡 full 数据正式记录。

### C1–C5 原生 DDP 验收（C6 前置条件）

- 1/2/4 卡正确性检查通过；
- strong 和 weak scaling 分开报告；
- 有 NCCL/AllReduce 与 GPU 拓扑证据；
- 能解释四卡没有达到四倍速度的原因；
- 有一次可恢复、可核对的完整训练窗口覆盖记录。

### C6. DeepSpeed ZeRO（完成 C1–C5 后实施）

#### C6.1 接入边界

1. 保留 `train.py` 作为原生 PyTorch/DDP 基线，新建 `train_deepspeed.py` 作为 DeepSpeed Engine 入口。
2. 复用 `model.py`、`ddp_windows.py`、TinyStories 数据和正式模型配置，不改变数据顺序和模型结构。
3. 在 `config/deepspeed/` 保存 ZeRO-0/1/2/3 和可选 offload 策略；运行时根据 world size 解析实际 micro-batch、梯度累积和全局 batch。
4. 将完整解析后的 DeepSpeed 配置写入每次结果，不只保存原始 JSON。
5. 第一轮保持 `compile=False`、BF16、相同 optimizer 超参数和相同全局 tokens/update，不同时引入 fused optimizer 或 offload。

#### C6.2 Engine 与更新正确性

1. 先用单卡 ZeRO-0 完成 forward、`engine.backward`、`engine.step` 和短训练冒烟检查。
2. 用相同初始权重和全局 batch，对照原生 DDP 与 ZeRO-0 的一次参数更新。
3. 核对各 rank 窗口、全局窗口哈希、loss、梯度累积边界和更新后参数误差。
4. 正确性对照通过后再启用 ZeRO-1/2/3，不直接从 ZeRO-3 开始排错。

#### C6.3 ZeRO 阶段对照

1. 在同一台四卡服务器上对照原生 DDP、ZeRO-0、ZeRO-1、ZeRO-2 和 ZeRO-3。
2. 使用相同模型、初始化、数据窗口、精度和全局 tokens/update，分别记录 1/2/4 卡条件下的结果。
3. 正式无 Profiler 基准至少预热 20 次、测量 100 次并独立重复 3 次。
4. 报告最慢 rank step 时间、全机 tokens/s、每卡峰值显存、loss、参数分片情况和 checkpoint 开销。
5. 单独采集通信日志与 PyTorch Profiler trace，观察 AllReduce、ReduceScatter、AllGather、bucket 和计算/通信重叠；不将开启通信计时后的速度当作正式吞吐。
6. 允许 ZeRO-3 在当前小模型上变慢，但必须用通信和显存证据解释原因。

#### C6.4 模型容量边界

1. 保持 GPT 模型家族、TinyStories 数据、序列长度和精度不变，逐步增加层数或宽度，不在本实验中更换为另一种模型架构。
2. 固定 micro-batch 后分别寻找 DDP、ZeRO-2 和 ZeRO-3 的最大可训练参数规模，记录成功与 OOM 边界。
3. 对每个容量点报告模型配置、参数量、每卡显存、step 时间、tokens/s 和 CPU 内存。
4. 只在无 offload 的 ZeRO-3 完成后，再将 optimizer/parameter offload 作为独立变量，记录容量收益和 PCIe/CPU/NVMe 代价。

#### C6.5 分布式 checkpoint

1. 由所有 rank 共同保存分片 checkpoint，包含 model、optimizer、update 计数和数据窗口进度。
2. 在新进程中恢复训练，检查下一个窗口、loss、optimizer state 和更新次数连续。
3. 将 ZeRO-2/3 checkpoint 合并为普通 FP32 `state_dict`，用非 DeepSpeed 的 `GPT` 加载并对照固定输入的输出。
4. 记录 checkpoint 分片数、总大小、保存/恢复时间和合并所需 CPU 内存。

#### C6 验收

- ZeRO-0 与原生 DDP 的更新正确性对照通过；
- ZeRO-0/1/2/3 有同条件吞吐、显存和通信对照；
- 能解释各 ZeRO 阶段分片的状态、新增的集合通信和性能取舍；
- 有 DDP OOM 但 ZeRO 可训练的容量边界证据；
- ZeRO 分片 checkpoint 可保存、恢复和合并为普通权重。

截至 2026-10-07，C6 已在四卡 RTX 4090 D 上验收。ZeRO-0/1/2/3 的 1/2/4 卡正式矩阵、通信 trace、保存恢复和 ZeRO-2/3 FP32 合并均已完成；所有 loss 有限且窗口哈希与计划一致。51.17M 模型的四卡吞吐分别为 357,031、334,070、289,498、152,417 token/s，原生 DDP 为 320,468 token/s。容量探测得到 DDP 6.94 亿成功/9.84 亿 OOM、ZeRO-2 20.41 亿成功/21.56 亿 OOM、ZeRO-3 22.74 亿成功/23.95 亿 OOM。结果说明小模型吞吐优先 ZeRO-0/1，大模型容量不足时再进入 ZeRO-2/3；详细证据与限制见 `docs/c_stage_results.md`。

### C7. PyTorch FSDP2 对照（可选）

进入条件：C6 完成且仍有时间和多卡资源。该阶段不阻塞后续 MoE 和 Triton 主线。

1. 仅对照原生 DDP、DeepSpeed ZeRO-3 和 FSDP2 `fully_shard`，不重复所有 ZeRO 阶段的完整矩阵。
2. 使用同一 GPT 模型、初始权重、数据窗口、全局 batch、BF16 和相同的无 Profiler 计时方法。
3. 选择两个模型点：当前 50.91M 性能基准，以及 C6.4 中接近 DDP OOM 的容量点。
4. 对比吞吐、峰值显存、AllGather/ReduceScatter 形态、包裹策略、checkpoint 格式和接入复杂度。
5. 验证 FSDP2 分片 checkpoint 保存与恢复，并与 ZeRO-3 的分片权重生命周期做概念对照。
6. 不为了增加框架数量再接入 Accelerate、Lightning 或 Colossal-AI；只在出现新的明确实验问题时扩展。

#### C7 验收

FSDP2 与 ZeRO-3 在同条件下都能正确训练和恢复；能用实测证据解释两者的显存、通信、性能和工程取舍。

截至 2026-10-07，C7 可选对照也已完成。FSDP2 的 1/2/4 卡正式矩阵、AllGather/ReduceScatter trace、四卡 DCP 保存恢复以及 9.84 亿参数 DDP-OOM 点均通过。51.17M 模型四卡吞吐为 237,046 token/s，是原生 DDP 的 74.0%。同一重开拓扑的 9.84 亿点上，FSDP2 为 22,481 token/s、12.091/13.898 GiB allocated/reserved，ZeRO-3 为 33,805 token/s、14.020/17.436 GiB，形成明确的 GPU 显存/吞吐取舍。实验没有继续寻找 FSDP2 的首个失败点，因此 9.84 亿是已测试成功下界，不是最大容量。

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

### D4. 编译兼容性与动态路由边界

1. 保持 D3 工作量和数据窗口不变，只切换 `torch.compile`。
2. Dense/MoE 各重复三次无 Profiler 正式计时。
3. 用短 trace 和 graph-break/recompile 日志解释编译收益。
4. 如果 MoE 没有收益，保留负结果，并把优化目标指向 dispatch 表示而不是泛化的“继续编译”。

### D 阶段验收

路由、输出和梯度正确；有 TinyStories 正式窗口上的 Dense/MoE 对照；能根据路由指标解释负载情况；能用无 Profiler A/B、trace 和 graph-break 证据说明编译为什么对 Dense 有效、对当前 MoE 无效。

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

截至 2026-10-08，E 阶段已完成。固定容量 dispatch 消除了 D4 的
`bincount/nonzero` 动态边界，compiled Triton MoE 达到 65,945
token/s，是 compiled Dense 的 96.92%。但 compiled
`torch.bmm` 也达到 65,917 token/s，说明端到端提升主要来自稳定数据
布局，而不是手写 GEMM 本身。完整协议与证据见
`docs/e_stage_protocol.md` 和 `results/e_triton_moe/README.md`。

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

## G. 从训练 checkpoint 到增量推理

G 阶段先建立 RouteScale 自身的推理基线，不直接从完整 serving 框架开始。Dense
checkpoint 是首个正式对象；MoE 必须等 Dense 的导出、logits 等价和 KV Cache
正确性全部通过后再进入。

### G1. 推理产物契约

1. 从训练 checkpoint 导出权重、`GPTConfig`、tokenizer 名称与版本、特殊 token、
   dtype 和源文件 SHA256；训练恢复所需的 optimizer/RNG/数据窗口状态不进入推理产物。
2. 清理 `torch.compile` 产生的参数名前缀，验证 tied embedding，并用新进程 strict load。
3. 固定一组短、中、长 prompt，在 FP32 和正式推理 dtype 下保存 reference logits。
4. 使用 greedy decoding 对照训练模型和导出模型；先定位首个不一致 token，再允许测性能。

### G2. 朴素生成与 KV Cache

1. 将当前每步重算完整上下文的 `GPT.generate` 作为朴素基线。
2. 为每层 attention 增加增量 Prefill/Decode 接口、position offset 和连续 KV Cache。
3. 分别验证一次性 Prefill、逐 token Decode、超过 block size 和多 batch 请求。
4. 对照完整重算与 cached decode 的 logits、greedy token 和结束条件。
5. 分开报告 Prefill latency、TTFT、Decode tokens/s、每 token latency 和 KV Cache 显存。

### G3. 推理正式协议

正式 workload 至少覆盖：

- batch/concurrency 为 1、8 和 32；
- prompt 长度为 128 和 512，输出长度为 64 和 256；
- 无共享前缀和高共享前缀两类请求；
- 预热后的稳态时间，以及单独记录的加载、编译和 CUDA Graph capture 时间。

每组至少三个独立进程。除 output tokens/s 外，同时记录 TTFT、ITL/TPOT、E2E
latency p50/p95/p99、峰值显存和成功请求数。发生 OOM 时必须记录请求形状、并发、
KV Cache 大小和失败阶段。

### G 阶段验收

训练 checkpoint 能在新进程中作为独立推理产物加载；朴素与 cached 路径在容差内
logits 等价且 greedy token 一致；能解释 Prefill 与 Decode 的计算/显存差异，并用
正式结果量化 KV Cache 的收益和代价。

---

## H. nano-vLLM：推理执行核心

nano-vLLM 用作“机制显微镜”。其上游源码保留在独立环境或 fork 中，RouteScale
只保存固定 commit、环境、必要 patch、统一 workload 和结果，不把上游包加入训练
环境的默认依赖。

### H1. 上游复现与源码路径

1. 固定 `GeeeekExplorer/nano-vllm` commit、许可证、Python/PyTorch/CUDA 和模型 revision。
2. 使用其原生支持的 Qwen3-0.6B 跑通官方离线生成，先复现功能，再测本机结果。
3. 按 `LLMEngine → Scheduler → Sequence → BlockManager → ModelRunner → Attention`
   顺序阅读并画出 Prefill、Decode、抢占、block 分配和释放的数据流。
4. 为 block 分配、引用计数、释放、Prefix Cache 命中和 cache exhaustion 写最小状态测试。

### H2. 单变量消融

依次对照：

1. eager 与 `torch.compile`/CUDA Graph；
2. Prefix Cache 关闭与开启；
3. 固定 batch 与连续批处理；
4. contiguous KV 与 paged/block KV 的占用、碎片和可服务并发；
5. 在多卡机器上对照 TP=1/2/4，并记录 collective 与计算时间。

不得把上游 README 的数字当作本项目结果。所有结论必须来自相同硬件、模型 revision、
请求集合、采样参数和测量边界。

### H3. RouteScale 模型接入

1. 在隔离 fork 或可重放 patch 中增加 Dense `RouteScaleGPT`，处理 learned positional
   embedding、LayerNorm、GELU、MHA、tied embedding 和 RouteScale 权重命名。
2. 先加载 G 阶段导出的 Dense 权重，不同时加入 MoE、量化或 speculative decoding。
3. 用 G 阶段 reference logits 和 greedy token 验证模型层、KV 写入和采样语义。
4. 对照 RouteScale native cached path 与 nano-vLLM 的吞吐、TTFT、显存和并发能力。

### H 阶段验收

能够脱离文档解释 Scheduler、BlockManager、ModelRunner 和 Paged KV Cache 的职责；
所有 block 生命周期测试通过；至少完成两项正式消融；RouteScale Dense checkpoint
在 nano-vLLM 接入路径上与 G 阶段 reference 等价，或保留定位清楚、可复现的阻塞记录。

---

## I. Mini-SGLang：在线 Serving 系统

Mini-SGLang 用作“系统放大镜”。本阶段研究从 HTTP 请求到流式输出的完整控制面，
不重复实现 H 阶段已经证明的基础 KV Cache。

执行顺序：先在本机单卡完成 I1 服务链路、I2 单卡消融和 I3 在线负载。
H2 与 I2 的 TP=2/4 等多卡实验统一留到后期租卡窗口；租卡前准备好配置、
smoke 与采集脚本，窗口内在同一台机器补跑 TP=1 基线。不以单卡结果代替多卡验收。

### I1. 服务链路与进程边界

1. 固定 `sgl-project/mini-sglang` commit、环境和 Qwen3-0.6B revision，使用独立
   `uv` 环境或容器启动 OpenAI-compatible server。
2. 画出 `API Server → Tokenizer → Scheduler → Engine → Detokenizer → Client`
   链路，标明 ZMQ 控制消息、GPU tensor 和多卡 NCCL 通信。
3. 区分请求排队、Prefill、Decode、detokenize 和网络返回时间，不只记录总 wall time。
4. 验证客户端断开、EOS、最大输出长度、服务关闭和请求资源释放。
5. 明确总接入上限、429 过载策略、worker 存活/readiness 和故障传播；
   分别验收客户端终态、后端停止、table/KV、frontend/detokenizer 引用及进程退出。

2026-10-31：本机 TP1、共享 tokenizer、Graph/Overlap 关闭的 I1 生命周期验收通过；
七个可重放补丁、四个真实 GPU 场景、130 项场景检查和 40 项本机测试见
[修复报告](../results/i_minisglang/i1_repaired/README.md)及
[源码说明与运行手册](i1_lifecycle_repairs.md)。I2 本机范围后续已验收；I3 与 TP2/4 仍待执行。

### I2. 调度与缓存消融

1. Naive Cache 对照 Radix Cache，分别使用无共享前缀和高共享前缀 workload；
2. Chunked Prefill 开关及不同 chunk size，观察长 prompt 对短请求 TTFT 的影响；
3. Overlap Scheduling 开关，检查吞吐收益是否伴随尾延迟变化；
4. CUDA Graph 开关和不同 capture batch bucket；
5. Attention backend 只在目标硬件实际支持时比较，不能用后端不可用时的 fallback
   结果声称性能差异；
6. 多卡对照 TP=1/2/4，并用 NCCL trace 解释小模型和较大模型的通信/计算比。

2026-10-31：I2 v1 的七配置 HTTP 生命周期检查通过（84 请求、280 项检查，另有
12 项 CPU 测试），含 Overlap 的在途取消扩展。首项全模型数值门槛未通过，
1024 个 Naive 向量全部有限但均超出冻结门槛，正式性能进程为 0/21。
[失败、隔离诊断与后续定位](../results/i_minisglang/i2_single_gpu_v1/README.md)保持可审计；
该 HF 参考范围尚未验收，不能把 HTTP 检查或诊断当成全模型 HF 等价通过。

后续独立范围已完成：首分叉定位到 FlashInfer 自动 split-KV 及 BF16 GEMM 行形状，
共同 prefill no-split + linear tile512 后，新seed/768-token输入及七配置共12544个
向量全部 exact；同策略 HTTP 84请求/280检查、21/21正式进程与160项性能审计通过。
见[根因说明](i2_numerical_findings.md)与[本机 FI/TP1 形状控制范围报告](../results/i_minisglang/i2_shape_performance_v1/README.md)。
结果包含额外填充成本和不确定性；原HF FP32失败保持不变，FA4/TRTLLM在当前锁定栈中
实际探针不可用，未使用fallback比较。TP2/4仍留待集中租卡。

用户追加优先核对原HF FP32门槛：[外部参考与精度方案验证](../results/i_minisglang/i2_external_alignment_v1/README.md)。
完整原批次复现后，首个外部分叉定位到layer0 RMSNorm舍入；HF自身BF16也未满足原0.5%门槛。
当时保留旧BF16失败，先单独验证提高精度的候选方案，并要求补齐调度与HTTP验收后进入I3，
不直接套用原BF16性能结果；完整验收见下段。
FP16主路径配合FP32残差及层间/最终norm，候选阶段在Naive/Radix合计2048个向量上满足原阈值，
最坏相对RMSE为0.322%；纯FP16在Radix仍有1个越界向量，其失败也保留。

新精度已扩展到全部七配置：12544/12544个混合精度向量通过原HF FP32门槛，
最坏相对RMSE为0.3614%；同期BF16内部对照12544向量exact，新增20项CPU检查通过。
此前Windows另一个Python实验占用92 MiB显存，第三轮HTTP被进程日志门槛拒绝；
用户释放GPU后，2026-11-01的http004独立通过两模式14配置、168请求、560检查，
关闭预算及无残留检查也通过。42/42正式性能进程、347项独立审计、完整图表/CSV
及最终验收全部完成，**I2本机FI/TP1已关闭，可开始I3；TP2/4仍后置**。
b3 shared warm中，mixed Graph[1,2,4]相对mixed Radix的配对吞吐中位为5.36倍，
保留同期BF16对照约95.3%吞吐；norm多kernel实现成本、范围与最大21.19%的进程CV
均明示，不能据此宣称通用最优或在线SLO收益。原BF16/HF失败和三轮HTTP失败保留。
见[最终工程报告](../results/i_minisglang/i2_precision_v1/README.md)与
[重试目录及独占GPU说明](i2_precision_retries.md)。

### I3. 在线负载协议

正式在线实验至少包含低请求率、接近饱和点和超过饱和点三档流量；报告 request
throughput、input/output token throughput、TTFT、ITL/TPOT、E2E p50/p95/p99、
排队时间、cache hit rate、抢占/重算和满足延迟目标的 goodput。离线吞吐与在线延迟
使用不同结果表，不互相替代。

### I 阶段验收

能够解释控制面与执行面的边界；有 Radix Cache、Chunked Prefill 和 Overlap
Scheduling 的单变量证据；能说明平均吞吐、TTFT 和尾延迟为什么可能朝不同方向变化；
多卡结果包含拓扑、NCCL 和每 rank 显存证据。

---

## J. 训练—推理联合复盘

### J1. 统一对照边界

1. 使用 Qwen3-0.6B 作为 nano-vLLM 与 Mini-SGLang 的共同原生模型，只比较双方都
   支持且语义一致的功能。
2. 使用 RouteScale Dense checkpoint 对照朴素生成、native KV Cache 和 nano-vLLM
   接入路径；Mini-SGLang 的 RouteScale 模型支持为可选扩展，不作为理解在线系统的
   前置条件。
3. 所有来源记录模型 revision、框架 commit、请求哈希、采样参数、warmup、测量时间、
   GPU 拓扑和环境；不同 GPU 或不同模型的绝对数字不进入同一排名。

### J2. MoE 的 train/serve gap

1. 比较训练阶段大量 token 的固定容量 dispatch 与 Decode 阶段每请求一个新 token
   的动态 batch，量化 expert 利用率和 padding 浪费。
2. 检查 Continuous Batching 是否让 Decode token batch 足以摊薄 router、pack/unpack
   和 grouped expert GEMM 开销。
3. Prefill 与 Decode 分开选择 dispatch/kernel，不默认复用 E 阶段的正式 shape。
4. 推理路径首先保证不因 capacity/drop 改变输出语义，再讨论静态 bucket、CUDA Graph
   或新的 grouped kernel。
5. 只有 Dense serving 已稳定且 Profiler 证明 MoE compute 是热点，才接入 RouteScale MoE。

### J3. 最终决策

最终报告必须回答：

- 哪些训练基础设施结论会随工作负载规模变化？
- nano-vLLM 的收益来自 KV 复用、调度、内存管理还是执行图？
- Mini-SGLang 的 Radix Cache、Chunked Prefill 和 Overlap Scheduling 分别改善哪类负载？
- 训练侧 fixed shape 与服务侧 batch bucket 的共同点和语义差异是什么？
- 对本地单卡、四卡实验机和在线服务，分别推荐哪条最简单且证据充分的路径？

### J 阶段验收

形成从训练、导出、离线推理到在线服务的完整证据链；共同模型和共同 workload 的
对照可复现；结论同时包含收益、负结果、适用规模、硬件限制和下一步决策。最终更新
README、学习笔记、结果索引、五分钟讲解和简历表述。

---

## 项目文件职责

| 路径 | 职责 |
|---|---|
| `docs/RouteScale_实施教程大纲.md` | 项目路线、任务和验收条件 |
| `docs/RouteScale_学习笔记.md` | 概念、问答、易错点、判断方法和复习题 |
| `docs/environment.md` | 环境和硬件版本 |
| `results/README.md` | 已完成实验的配置、原始值和结论 |
| `results/` | 可审查、适合提交 Git 的 JSON/TXT/Markdown 指标与报告，不保存 checkpoint 或大型 trace |
| `artifacts/` | 已完成远端实验的冻结证据快照和 manifest |
| `runs/checkpoints/` | 本机可变 checkpoint、optimizer state 和分布式分片，不提交 Git |
| `runs/traces/` | 本机 Profiler trace 与运行日志，不提交 Git |
| `runs/archives/` | 本机证据压缩包和 Git bundle，不提交 Git |
| `notebooks/` | 模型 sizing 与 scaling-law notebook |
| `data/tinystories/prepare.py` | debug/full 数据转换 |
| `data/tinystories_debug/manifest.json` | 调试数据清单 |
| `data/tinystories_full/manifest.json` | 完整数据清单 |
| `config/train_tinystories_debug.py` | 数据和训练链路排错配置 |
| `config/train_tinystories.py` | TinyStories 训练配置 |
| `model.py` | GPT 模型与后续模型接入点 |
| `train.py` | 训练、计时、Profiler 和 DDP 主入口 |
| `distributed_common.py` | 三种分布式后端共享的数据、模型、optimizer 与评估语义 |
| `train_coverage.py` | C5 确定性完整覆盖与原生 DDP checkpoint 入口 |
| `train_deepspeed.py` | C6 阶段建立的 DeepSpeed Engine 主入口 |
| `train_fsdp2.py` | C7 阶段建立的 FSDP2 `fully_shard` 与 DCP 入口 |
| `config/deepspeed/` | C6 阶段建立的 ZeRO 与 offload 策略 |
| `scripts/preflight_multigpu.py` | 目标机器 GPU/软件/数据/NCCL collective 准入检查 |
| `docs/c_stage_multigpu_runbook.md` | 四卡租用规格、准入顺序和 smoke 命令 |
| `results/c5_coverage/` | C5 覆盖率、完整 validation 与恢复正确性结果 |
| `results/c6_deepspeed/` | C6 DeepSpeed 正确性、性能、容量、通信与 checkpoint 指标 |
| `results/c7_fsdp2/` | C7 FSDP2 正确性、性能、通信与 DCP 结果 |
| `moe.py` | D 阶段建立的 Top-1 MoE |
| `serving/export_checkpoint.py` | G 阶段训练 checkpoint 到推理产物的导出与校验 |
| `serving/native/` | RouteScale 朴素生成、增量模型、KV Cache 与最小调度实现 |
| `serving/backends/` | nano-vLLM 离线 adapter 与 Mini-SGLang HTTP adapter |
| `serving/workloads.py` | 离线/在线请求集合、共享前缀比例和 workload 哈希 |
| `serving/metrics.py` | TTFT、ITL/TPOT、E2E、goodput、显存与 cache 指标 |
| `config/serving/` | G–J 阶段固定的模型、请求、缓存、调度和并行配置 |
| `scripts/benchmark_offline_serving.py` | RouteScale/nano-vLLM 离线统一测量入口 |
| `scripts/benchmark_online_serving.py` | Mini-SGLang 在线请求回放与延迟统计入口 |
| `third_party/serving_sources.lock` | 上游 URL、commit、模型 revision、许可证和隔离环境记录 |
| `results/g_serving/` | G–J 阶段的正确性、离线/在线性能指标和结论；原始 trace 写入 `runs/traces/` |
