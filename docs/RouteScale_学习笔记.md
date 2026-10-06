# RouteScale 学习笔记与实验判断手册

> 更新日期：2026-10-06
> 这份文档整理 RouteScale 学习过程中已经讨论并实际验证的知识。它不是聊天记录，也不是实施计划：实施顺序和验收条件见 [`RouteScale_实施教程大纲.md`](RouteScale_实施教程大纲.md)，实验原始结果见 [`../results/README.md`](../results/README.md)。

## 0. 怎样使用这份笔记

每一章都尽量回答四件事：

1. **它是什么：**先建立准确、简单的心智模型。
2. **为什么有用：**说明它与训练正确性、显存或性能的关系。
3. **怎样判断：**给出公式、现象或实验方法，而不是只记结论。
4. **容易错在哪里：**记录学习过程中真实出现过的混淆点。

复习时建议先遮住“答案”，尝试口述章节末尾的自测题。能结合本项目的数字解释，才算真正掌握；只认得名词还不够。

文档对学习状态采用四种证据：

| 状态 | 含义 |
|---|---|
| 已复述 | 已经在交流中用自己的话解释过，错误得到过纠正 |
| 已实测 | 已经通过代码、日志或文件校验得到证据 |
| 初步理解 | 已经解释过，但还没有闭卷复述或独立实验 |
| 后续验证 | 要在 Profiler、DDP、MoE 或 Triton 阶段亲自完成 |

### 掌握层次

| 层次 | 判断标准 | 当前应达到的范围 |
|---|---|---|
| 会复述 | 能用自己的话说清输入、输出和作用 | tokenizer、batch、Block、loss、梯度 |
| 会计算 | 能代入实际配置算 shape、tokens/update、扩展效率 | A–C 阶段核心要求 |
| 会诊断 | 看到 OOM、吞吐停滞或多卡低效率，知道先测什么 | B–C 阶段逐步形成 |
| 会实现 | 能修改代码并验证正确性 | 采样器、Profiler、DDP、MoE、Triton 分阶段完成 |

### 内容架构

| 部分 | 章节 | 复习目标 |
|---|---|---|
| 训练心智模型 | 1–6 | 能从原始故事讲到一次参数更新 |
| 配置与系统判断 | 7–11 | 能解释显存、配置选择、优化目标和性能现象 |
| 可靠性能实验 | 12 | 能解释 B1/B2 测量、诊断和验证闭环 |
| 多卡实施规划 | 13 | 能区分 DDP 正确性、strong/weak scaling 和通信分析 |
| 诊断与复习 | 14–16 | 按现象选择诊断动作，区分真正掌握与初步理解 |

建议第一次按顺序阅读；以后遇到具体问题，直接查第 14 章诊断表，再回到对应原理章节。

---

## 1. 一次训练更新的完整地图

先记住主路径：

```mermaid
flowchart LR
    A[原始文本] --> B[Tokenizer]
    B --> C[token ID 文件]
    C --> D[get_batch: X 和 Y]
    D --> E[Embedding]
    E --> F[Transformer Blocks]
    F --> G[logits]
    G --> H[与 Y 比较得到 loss]
    H --> I[backward 计算梯度]
    I --> J[optimizer.step 更新参数]
    J --> K[清空梯度]
```

一次**前向传播**是从 `X` 计算 logits 和 loss。一次**参数更新**还包括反向传播和优化器更新。使用梯度累积时，一次参数更新会包含多次前向与反向。

先区分两个层次：

```text
一次 micro-step
= 一个 micro-batch 的 forward + backward

一次 optimizer update
= 多个 micro-step 累积梯度
+ gradient clipping
+ optimizer.step
+ zero_grad
```

当前配置的 `gradient_accumulation=4` 表示四次 forward/backward 后才执行一次参数更新，不是连续更新四次参数。

### 当前配置中一次更新发生了什么

```text
micro-batch = 8
sequence length = 512
gradient accumulation = 4
GPU 数 = 1
```

因此：

```text
一个 micro-batch 的训练位置数
= 8 条序列 × 每条 512 个位置
= 4,096

一次 optimizer update 的训练位置数
= 4,096 × 4 次梯度累积
= 16,384
```

这里的 16,384 通常简称为一次更新处理的 token 数。严格来说，它是参与下一个 token 预测的训练位置数。

`n_layer`、`n_head` 和 `n_embd` 不出现在这个 token 数公式里。它们决定“处理每个 token 要做多少计算”；`batch_size`、`block_size` 和梯度累积决定“一次参数更新处理多少训练位置”。

### 易错点

- 前向传播不等于完整参数更新；它还没有执行 `backward()` 和 `optimizer.step()`。
- 四次梯度累积不是四次参数更新；四次 backward 只产生一次 `optimizer.step()`。
- 一条序列不是只预测最后一个 token。训练时每个位置都预测它的下一个 token。
- loss 是一个汇总后的数，但它来自当前 micro-batch 中所有预测位置。

---

## 2. 数据怎样从故事变成本地训练文件

### 2.1 Hugging Face 在这里扮演什么角色

TinyStories 托管在 Hugging Face。数据准备阶段从远端读取它；性能训练阶段使用已经生成的本地文件：

```text
Hugging Face 上的故事
→ 流式读取文本
→ GPT-2 tokenizer 编码
→ 每篇故事追加 EOT
→ token 缓冲区
→ uint16 增量写盘
→ 本地 train.bin / val.bin
```

所以“使用 Hugging Face 数据集”不等于训练时一直从云端读取。RouteScale 的正式训练读取本地 `memmap`，避免网络速度干扰训练性能。

### 2.2 Tokenizer 不是 batch 切分器

Tokenizer 的任务是把文本转换成 token ID：

```text
文本 → token 片段 → 整数 ID
```

GPT-2 使用 byte-level BPE。一个 token 可能是：

- 一个常见单词；
- 单词的一部分；
- 标点或空格组合；
- 退化到字节表示的罕见文本。

Tokenizer 不负责决定 `batch_size`，也不负责切出 `block_size`。batch 和 block 是训练阶段的 `get_batch()` 从 token 文件中取出来的。

### 2.3 EOT 的作用

EOT 是 End Of Text。脚本在每篇故事后追加 GPT-2 的 EOT token：

```text
故事1 token ... EOT 故事2 token ... EOT
```

它告诉模型一篇故事已经结束。它是 token 流中的边界标志，不会把 `.bin` 物理切成很多文件。随机取得的一个训练 block 可以跨过 EOT，此时模型能看到“上一篇结束，下一篇开始”。

### 2.4 为什么磁盘使用 `uint16`，训练时转换为 `int64`

一个整数所需空间由位数决定：

```text
uint16 = 16 bit = 2 byte
int64  = 64 bit = 8 byte
```

因此同样数量的 token ID：

```text
uint16 文件大小 = int64 文件大小的 1/4
```

GPT-2 词表有 50,257 个 token，最大 ID 小于 65,536，所以能安全存进 `uint16`。但 PyTorch 的 embedding 查表需要 `torch.long`，也就是 `int64` 索引，因此只把当前 batch 转成 `int64`：

```text
磁盘和 memmap：uint16，节省存储和 I/O
当前训练 batch：int64，满足 embedding 接口
```

这不是“训练数值精度”的 BF16/FP32；它只是 token 索引的数据类型。

### 2.5 为什么使用流式处理和缓冲区

完整 TinyStories 很大。如果先把全部文本和全部 token 放进内存，内存使用会随着数据规模增长，可能直接耗尽。

当前脚本采用：

```text
远端取少量故事
→ 编码
→ token 暂存在有限缓冲区
→ 缓冲区达到阈值后写入磁盘
→ 清空缓冲区
→ 继续下一批
```

假设完整数据有 4.74 亿 token，Python 整数在内存里的实际开销远大于 2 字节。流式处理让内存主要取决于缓冲区大小，而不是数据集总大小。

它与以后 DDP 的关系是：数据系统必须持续向 GPU 提供数据，不能要求单机内存装下完整数据。但 DDP 的 rank 分片、避免重复和确定性顺序是训练采样器的职责，流式下载本身不会自动解决这些问题。

### 2.6 train/validation split 为什么保持不变

- `train` 用来更新模型参数。
- `validation` 只用来估计模型在未参与训练的数据上的 loss。

项目直接使用数据集已有的 split，并固定数据 revision。所有对照实验使用相同 token 文件，避免数据变化被误判为模型或系统优化效果。

### 2.7 SHA256 有什么用

SHA256 可以看作文件内容的指纹。两台机器上对应文件的 SHA256 相同，说明它们的字节内容以极高概率完全一致。

它能够证明：

- 本地和租用服务器使用相同 token 文件；
- 复制过程没有损坏文件；
- 实验数据能够复现。

它不能保证每个 GPU 读取不同的数据。多卡窗口怎样分给 rank，需要单独设计采样计划。

### 当前数据证据

| 文件 | 故事数 | token 数 | 大小 |
|---|---:|---:|---:|
| full train | 2,119,719 | 473,992,236 | 947,984,472 bytes |
| full validation | 21,990 | 4,765,918 | 9,531,836 bytes |

文件大小满足：

```text
字节数 = token 数 × 2
```

这与 `uint16` 每个 token 占 2 字节一致。

---

## 3. `get_batch()`：batch、block、X 和 Y

核心代码：

```python
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([
    torch.from_numpy(data[i:i+block_size].astype(np.int64))
    for i in ix
])
y = torch.stack([
    torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
    for i in ix
])
```

### 3.1 `block_size` 是什么

`block_size` 是每条输入序列包含多少个连续 token。

若 token 流是：

```text
A B C D E F ...
```

且 `block_size=4`，从 A 开始取得：

```text
X = A B C D
Y = B C D E
```

模型分别学习：

```text
看到 A       → 预测 B
看到 A B     → 预测 C
看到 A B C   → 预测 D
看到 A B C D → 预测 E
```

所以长度为 4 的 block 产生 4 个训练位置。

### 3.2 `batch_size` 是什么

`batch_size` 是一次前向和反向并行处理多少条 block。

如果：

```text
batch_size = 8
block_size = 512
```

那么 `X` 和 `Y` 的 shape 都是：

```text
(8, 512)
```

第一个维度是 8 条序列，第二个维度是每条序列的 512 个位置。

### 3.3 原版采样为什么不保证遍历完整数据

`torch.randint()` 随机选择起点，而且是有放回采样：

- 同一个起点可能多次抽到；
- 某些起点可能一直没有抽到；
- 训练 300 步不等于按顺序读完 300 个不重复区域。

当前候选配置 300 次更新处理的训练位置数是：

```text
300 × 16,384 = 4,915,200
```

这远少于 473,992,236 个源 token，而且还可能存在重复采样。因此 300 步只能证明训练链路稳定，不能说完整遍历了数据。

正式的完整覆盖实验需要生成确定的窗口序列，再按 rank 分配，记录重复、丢弃和尾部处理。

### 3.4 为什么训练可以随机取样，而不是每次都严格遍历

训练目标是用许多样本估计“怎样修改参数能降低整体 loss”。随机 mini-batch 给出完整数据梯度的近似，连续更新能够逐渐学习总体规律，不要求每次更新都先看完全部数据。

严格遍历一轮也不自动得到最好效果：

- 训练可能进行一轮、多轮，或直接按固定 token 预算安排；
- 样本顺序会影响随机优化路径；
- 模型容量、学习率和训练时间同样决定效果；
- 看完每个样本一次，既不保证已经学会，也不保证不会过拟合。

原版随机有放回采样适合快速训练和功能调试，但不方便证明覆盖率。RouteScale 后面的四卡完整覆盖实验改用确定性窗口，是为了让“每个窗口处理过一次”可以核验，也是为了让 1/2/4 卡实验消费相同的全局工作，而不是因为随机训练方法本身无效。

### 易错点：两种“Block”

| 名称 | 含义 | 当前示例 |
|---|---|---|
| data block / `block_size` | 从 token 文件截取的一段输入 | 每段 512 token |
| Transformer Block | 模型内部的一层处理模块 | 模型共有 8 层 |

同一段 512-token 输入会依次经过 8 个参数不同的 Transformer Block。

---

## 4. token ID 怎样变成模型能够处理的向量

token ID 只是整数编号，没有天然的语义距离。模型使用 embedding 表查出向量：

```python
tok_emb = self.transformer.wte(idx)
```

如果：

```text
idx shape = (8, 512)
n_embd = 512
```

查表后的 shape 是：

```text
(8, 512, 512)
```

三个维度分别表示：

```text
8 条序列 × 每条 512 个位置 × 每个 token 512 个特征
```

Embedding 表开始时随机初始化，训练通过梯度逐渐调整。并不是人为给每个 token 编写固定编码。

模型还会给每个位置查一个位置向量：

```python
pos_emb = self.transformer.wpe(pos)
x = tok_emb + pos_emb
```

因此送入 Transformer Blocks 的每个位置同时包含：

- “这个 token 是什么”的可训练表示；
- “它处在序列第几个位置”的可训练表示。

注意：`n_embd=512` 表示向量有 512 个数，不是“512 层”。

---

## 5. Transformer Block 在做什么

nanoGPT 中一个 Block 的主结构是：

```text
LayerNorm
→ Causal Self-Attention
→ 残差连接
→ LayerNorm
→ MLP
→ 残差连接
```

可以用“交流后独立思考”理解：

- **Attention：**每个位置从自己以及前面的 token 收集相关信息。
- **MLP：**每个位置拿着已经汇总的信息，独立做更复杂的非线性变换。

### 5.1 Causal Self-Attention

`causal` 表示当前位置不能看到未来 token。训练位置 `C` 可以利用 `A B C`，不能偷看目标中的 `D`。

多头注意力把 512 维拆成多个子空间。当前配置：

```text
n_embd = 512
n_head = 8
每个 head 的宽度 = 512 / 8 = 64
```

不同 head 可以学习关注不同关系，最后再合并回 512 维。

### 5.2 MLP 做的“再变换”是什么

nanoGPT 的 MLP 对每个位置执行近似这样的变换：

```text
512 → 2048 → GELU → 512
```

第一层把特征扩展到更宽的空间，GELU 加入非线性，第二层压回模型宽度。它不是再去读取其他 token；跨位置的信息已经由 Attention 汇入当前位置。

### 5.3 为什么重复 8 个 Block

一个 Block 只能完成一轮信息交流和特征变换。多层允许模型逐步组合关系：

```text
较浅层：局部词形、简单搭配
中间层：句子关系、指代和模式
较深层：更组合化的上下文表示
```

这是直观解释，不代表每层必然具有固定职责。每个 Block 结构相似，但参数各自独立。

输入和输出 shape 通常保持 `(8, 512, 512)`，并不表示里面的数没变。每层都改变了向量所表达的信息。

### 5.4 残差连接和 LayerNorm

- 残差连接把模块输出加回原输入，帮助信息和梯度跨层传播。
- LayerNorm 控制每个位置特征的尺度，使训练更稳定。

当前阶段需要理解作用和 shape；Attention 数学推导、Flash Attention 内核和 LayerNorm 自定义 backward 可在 Profiler/Triton 阶段按需要深入。

---

## 6. logits、loss、梯度和模型参数

### 6.1 logits 是什么

最后的线性层把每个位置的 512 维向量映射到整个词表：

```text
(batch, sequence, 512)
→ (batch, sequence, vocab_size)
```

每个词表项得到一个分数，这些分数叫 logits。它们还不是概率；softmax 后才形成概率分布。

### 6.2 loss 是什么

每个位置都有一个正确答案 `Y`。交叉熵衡量正确 token 的预测概率是否足够高，再对所有位置汇总。

loss 下降通常说明模型在当前任务上学到了规律，但短期训练 loss 或少量随机 validation batch 不能单独证明模型最终质量。

### 6.3 模型参数是什么

模型参数是训练过程中会被优化器更新的数，包括：

- token embedding 和位置 embedding；
- Attention 的 Q、K、V 和输出投影矩阵；
- MLP 的两层权重和偏置；
- LayerNorm 的缩放与偏移；
- 最终输出层权重。

当前候选模型约有 50.91M 参数。输入 token ID、训练文本和标签 `Y` 都不是模型参数，不会被梯度更新。

### 6.4 梯度是什么

梯度回答：

> 某个参数稍微增大时，loss 会朝哪个方向变化、变化多快？

训练顺序是：

```text
forward 得到 loss
→ backward 计算每个参数的梯度
→ optimizer.step 根据梯度更新参数
→ zero_grad 清空旧梯度
```

AdamW 不只是简单减去梯度；它还保存梯度的一阶和二阶移动统计。现阶段需要知道这些状态占显存、优化器在更新参数，不要求逐行推导 AdamW。

### 6.5 梯度累积

梯度累积让多个 micro-batch 共享一次 `optimizer.step()`：

```text
micro-batch 1：backward，保留梯度
micro-batch 2：backward，继续累积
...
累积完成：optimizer.step
```

单卡公式：

```text
tokens/update = micro_batch × sequence_length × accumulation_steps
```

DDP 公式：

```text
global tokens/update
= world_size × micro_batch × sequence_length × accumulation_steps_per_rank
```

梯度累积主要用更多串行计算换较低的峰值激活显存。它不会像直接增大 micro-batch 那样同时把所有样本放进 GPU。

---

## 7. 显存花在哪里

训练时的主要显存组成：

| 内容 | 作用 | 受什么影响较大 |
|---|---|---|
| 模型参数 | 保存可训练权重 | `n_layer`、`n_embd` |
| 梯度 | 保存每个参数的导数 | 参数量 |
| 优化器状态 | AdamW 的移动统计等 | 参数量、状态精度 |
| 中间激活 | backward 需要的前向结果 | batch、序列长度、层数、宽度 |
| 临时工作区 | CUDA 算子计算使用 | 算子和输入 shape |
| PyTorch 缓存 | 加速之后的内存分配 | 历史峰值和分配策略 |

### 7.1 allocated 与 reserved

- `allocated`：当前张量实际占用的 CUDA 内存。
- `reserved`：PyTorch 向 CUDA 申请并保留的内存池，其中可以包含暂时空闲、等待复用的部分。

所以常见：

```text
reserved > allocated
```

当前 batch 8 实验峰值约为：

```text
allocated = 3899 MiB
reserved  = 5148 MiB
GPU total = 8151 MiB
```

### 7.2 OOM 是什么

OOM 是 Out Of Memory。GPU 无法为下一次参数、激活或临时工作区分配足够显存时，通常会出现：

```text
torch.OutOfMemoryError: CUDA out of memory
```

常用处理顺序：

1. 降低 micro-batch；
2. 增加梯度累积以保持 `tokens/update`；
3. 使用 BF16/FP16；
4. 缩短序列；
5. 使用激活检查点或更省内存的 Attention；
6. 减少模型层数或宽度；
7. 大模型多卡时使用参数、梯度、优化器状态分片。

例如从：

```text
batch 8 × accumulation 4
```

改成：

```text
batch 4 × accumulation 8
```

可以保持同样的 16,384 tokens/update，但需要更多串行 micro-step，速度可能下降。

---

## 8. 各配置项怎样影响模型、计算和显存

可以把模型想成一座“故事加工厂”：

| 参数 | 形象理解 | 当前配置的技术含义 |
|---|---|---|
| `n_layer=8` | 8 层连续加工车间 | 每段输入依次经过 8 个 Transformer Block |
| `n_head=8` | 每层有 8 位观察员，从不同角度看上下文 | 512 维在每层 Attention 中切成 8 个 64 维 head，最后合并回 512 维 |
| `n_embd=512` | 每个 token 有一张包含 512 项特征的档案 | 每个位置使用 512 维向量表示 |
| `block_size=512` | 每条送入工厂的故事片段有 512 个位置 | 每条训练序列产生 512 个下一个 token 预测位置 |
| `batch_size=8` | 一辆车一次运来 8 段故事 | 每个 micro-batch 同时处理 8 条序列 |
| `gradient_accumulation=4` | 收集 4 车货物的意见后再统一调整机器 | 四次 forward/backward 累积梯度，只执行一次参数更新 |
| `dtype=BF16` | 用较短的数字格式加工 | 主要计算使用 16 bit 格式，通常比 FP32 更省显存并适合 Tensor Core |
| `compile=False` | 按原始工序逐项执行 | 不启用图编译和由此产生的算子融合、编译预热等变量 |
| `dropout=0` | 不随机遮住任何工位 | 训练时关闭随机丢弃，使基线更简单且没有这项随机性 |

这组参数分成两类：

```text
决定每个 token 计算有多重：n_layer、n_head、n_embd
决定一次更新处理多少位置：batch_size、block_size、gradient_accumulation
影响怎样执行这些计算：dtype、compile、dropout
```

### 8.1 `n_layer`：Transformer Block 数量

增加层数通常会近似线性增加：

- Block 参数量；
- 中间激活；
- 前向和反向计算。

它改变模型本身，做系统 A/B 对照时通常固定。

### 8.2 `n_embd`：向量宽度

主要权重矩阵的参数量近似与 `n_embd²` 成正比。宽度从 512 增至 1024，许多矩阵的参数量接近 4 倍。

它同时影响参数、激活和矩阵乘法规模，是非常敏感的模型旋钮。

### 8.3 `n_head`：Attention 头数

当 `n_embd` 固定时，改变头数主要改变怎样切分向量：

```text
head_dim = n_embd / n_head
```

它通常不会让主要投影矩阵参数量随头数成倍增长，但会影响内核 shape、表达方式和执行效率。`n_embd` 必须能被 `n_head` 整除。

### 8.4 `block_size`：上下文长度

- 激活通常随序列长度明显增长；
- 普通 Attention 的关系矩阵和计算量近似按 `T²` 增长；
- 从 512 增至 1024，Attention 相关工作可能接近 4 倍。

### 8.5 `batch_size`：micro-batch 中的序列数

增加 batch 通常近似线性增加激活显存。较大的 batch 可能让 GPU 工作更饱满，但超过饱和点后吞吐不再明显增加。

### 8.6 BF16、FP16 和 FP32

```text
BF16/FP16：每个元素 2 byte
FP32：每个元素 4 byte
```

BF16 具有接近 FP32 的指数范围，通常比 FP16 更不容易溢出；FP16 训练常配合 GradScaler。混合精度不表示所有状态都变为 16 bit，master 参数或 AdamW 状态可能仍用 FP32。

### 8.7 `compile`

`torch.compile` 会捕获计算图并生成优化代码。它可能融合算子、减少 Python 和 kernel launch 开销，但首次编译很慢，也可能因 shape 改变重新编译。

因此分别记录：

- 启动/编译时间；
- 预热后的稳态 step 时间；
- 总训练时间；
- 是否出现 graph break 或重新编译。

### 8.8 `dropout`

Dropout 会在训练期间随机把部分中间值置零，用于正则化。`dropout=0` 表示本项目当前基线不执行这种随机遮挡。

关闭 Dropout 不代表所有训练都应该关闭它；这里是因为 TinyStories 系统实验首先关心稳定、可解释的性能对照。以后如果研究模型泛化，需要把 Dropout 当作训练超参数单独评估。

### 易错点

- `n_layer=8` 是前后串行的 8 个 Block；`n_head=8` 是每一层 Attention 内并行的 8 个视角。
- `n_head=8` 不会把 512 维变成 `512 × 8`；当前每个 head 是 `512 / 8 = 64` 维。
- `batch_size=8` 是一个 micro-batch，不是一次 optimizer update 的全部数据。
- `block_size=512` 是数据序列长度，不是 Transformer 层数。
- `compile=False` 不表示关闭 CUDA，只表示不使用 `torch.compile`。

---

## 9. 当前 50.91M 配置是怎样选出来的

当前本地候选：

```text
Transformer Blocks：8
Attention heads：8
向量宽度：512
序列长度：512
micro-batch：8
梯度累积：4
精度：BF16
compile：False
参数量：约 50.91M
```

它不是被证明“全局最优”的模型，而是当前 8 GiB GPU 上满足以下条件的本地基线：

- 规模足够形成有意义的 GPU 负载；
- 能稳定前向、反向、评估和保存；
- loss 正常下降；
- 显存保留安全余量；
- 后面可以用于学习 Profiler 和为 DDP 做准备。

### 9.1 决策顺序

1. **先确定任务：**当前目标是系统基线和稳定性，不是搜索最佳生成质量。
2. **选择适中的模型：**8 层、宽度 512 得到约 50.91M 参数；8 heads 对应 `head_dim=64`。
3. **固定有意义的序列长度：**使用 512 token，兼顾上下文和本地显存。
4. **逐步测试 micro-batch：**实际测试 4、6、8，而不是只凭猜测。
5. **选择安全可复现的点：**batch 8 没有 OOM，reserved 约占总显存 63%。
6. **用累积形成合理更新规模：**4 次累积得到 16,384 tokens/update。
7. **先固定 BF16、关闭 compile：**减少变量，先验证正确性和稳定性。
8. **跑 300 次精确更新并恢复 checkpoint：**确认不是只能运行几步。

### 9.2 实测选择证据

| micro-batch | tokens/update | 峰值 allocated | 峰值 reserved | 诊断 step 时间 |
|---:|---:|---:|---:|---:|
| 4 | 8,192 | 2426.3 MiB | 2772.0 MiB | 约 184 ms |
| 6 | 12,288 | 3157.7 MiB | 3968.0 MiB | 约 264 ms |
| 8 | 16,384 | 3899.2 MiB | 5148.0 MiB | 约 346 ms |

这些时间还不是正式 CUDA 基准，因为没有统一同步边界；它们只用于本地适配判断。正式比较需要 B 阶段的预热、CUDA 计时和重复实验。

### 9.3 稳定性证据

| step | validation loss 估计 |
|---:|---:|
| 0 | 10.9094 |
| 100 | 4.0805 |
| 200 | 3.7706 |
| 300 | 3.5545 |

运行中没有 NaN/OOM，checkpoint 成功恢复。这证明该配置能稳定学习，但 300 步和 20 个随机 validation batch 不足以证明最终模型质量。

### 9.4 换到四卡服务器后为什么还要重新确认

不同 GPU 的显存、Tensor Core、带宽和卡间连接都不同。本地配置可以作为起点，但正式 B–E 对照必须在目标机器上重新测 micro-batch 和稳态速度，再冻结配置。

不能用笔记本单卡吞吐和服务器四卡吞吐直接计算扩展效率；1/2/4 卡需要在同一台四卡机器、相同软件环境下测量。

---

## 10. 先明确优化目标，再调整参数

### 10.1 最大吞吐量

目标：单位时间处理最多 token。

```text
tokens/s = 测量窗口内处理的 token / 测量时间
```

初步策略：

- 逐步增大 micro-batch，直到吞吐趋于平坦；
- 使用 BF16 和适合硬件的矩阵维度；
- 保证数据供应跟得上 GPU；
- 尝试 `torch.compile`、算子融合和多卡；
- 用 profiler 判断计算、显存带宽、数据或 launch 瓶颈。

### 10.2 最低单步延迟

目标：尽快完成一次请求或一次 step。

初步策略：

- 减小 batch、序列或模型；
- 减少不必要的同步和 CPU 开销；
- 使用编译或融合降低 launch 开销。

较小 batch 可能降低延迟，却不能充分使用 GPU，所以吞吐量可能下降。延迟和吞吐量要分别报告。

### 10.3 最低显存

初步策略：

- 减小 micro-batch并使用梯度累积；
- 使用 BF16/FP16；
- 激活检查点用重新计算换显存；
- 使用更省显存的 Attention；
- 大模型使用 FSDP/ZeRO 等分片方法。

### 10.4 最快达到目标 loss

目标是 time-to-loss，例如“多少分钟到 validation loss 3.5”。

吞吐量更高不一定更快达到目标 loss。过大的全局 batch 或不合适的学习率可能降低样本效率。需要同时记录：

- 墙钟时间；
- 已处理 token；
- validation loss；
- 学习率和全局 batch。

### 10.5 DDP 扩展效率

目标：增加 GPU 后，总吞吐接近成比例增加。

```text
speedup = N 卡吞吐 / 1 卡吞吐
efficiency = speedup / N
```

初步策略：

- 在同一台同型号多卡机器上比较；
- 保持全局工作量或每卡工作量的定义清楚；
- 让每卡有足够计算，避免通信占比过高；
- 检查 AllReduce、负载不均和 GPU 拓扑；
- 尽量让梯度通信与反向计算重叠。

### 10.6 更大的模型

先通过减小 micro-batch、混合精度、梯度累积和激活检查点腾出显存，再考虑 FSDP/ZeRO、张量并行或流水线并行。

### 10.7 更长的上下文

增加 `block_size` 前先估计 Attention 的平方增长。通常需要降低 micro-batch、使用 Flash Attention、混合精度和激活检查点，并确认位置编码支持目标长度。

---

## 11. 常见性能现象背后的逻辑

### 11.1 为什么 eval 和保存让某些 step 突然很慢

普通训练 step 只包含训练工作。评估点可能额外执行多个 train/validation forward，保存点还要把大 checkpoint 写入磁盘。

```text
普通 step：forward + backward + optimizer
评估 step：普通工作 + 多个 eval forward + checkpoint I/O
```

所以评估/保存所在 step 变慢，不等于训练内核突然退化。正式吞吐计时应排除 validation、checkpoint、下载和 profiler 开销。

### 11.2 CUDA 异步与同步边界

CPU 提交 CUDA 工作后可以立即继续，不会默认等待 GPU 完成。只用普通 CPU 时钟包住一条 CUDA 调用，可能测到的是提交时间。

正式计时可在完整测量窗口前后使用同步，或使用 CUDA Event：

```text
同步 → 开始计时 → 多个训练 step → 同步 → 停止计时
```

不能在每个小算子后都同步，否则会破坏 CPU/GPU 流水和计算通信重叠。同步用来建立准确的测量边界，不应随意插满训练路径。

### 11.3 为什么增大 batch 后吞吐不再提升

小 batch 时，GPU 可能没有足够并行工作。增加 batch 能提高计算单元利用率。但达到计算能力、显存带宽或其他瓶颈后，继续增加 batch 只会提高显存和单步延迟，tokens/s 基本不变。

这时应停止盲目增加 batch，并用 profiler 区分：

- 矩阵计算是否已经饱和；
- 是否受显存带宽限制；
- 是否有大量小 kernel 和 launch 开销；
- CPU 数据准备是否让 GPU 等待；
- 是否存在数据拷贝或同步空洞。

### 11.4 多卡为什么可能远不到 N 倍速度

DDP 中每张 GPU 有完整模型副本，读取不同数据，独立前向和反向。反向过程中通过 AllReduce 汇总梯度，使每个 rank 得到一致的梯度并更新出一致参数。

如果每卡计算很少而梯度通信固定较大：

```text
计算 10 ms + 通信 8 ms
```

通信会占很大比例。若每卡计算增至 100 ms、通信约 15 ms，通信占比更低，更可能取得良好扩展。

还要检查：

- GPU 间使用 PCIe、NVLink 还是跨节点网络；
- 最慢 rank 是否拖住其他 rank；
- 数据准备是否不均；
- 通信是否能和 backward 重叠；
- micro-batch 是否小到无法形成足够计算。

### 11.5 `compile=True` 首次运行为什么很慢

首次运行可能经历图捕获、分析、代码生成和 CUDA 内核编译。假设编译花 30 秒，稳态每步节省 0.1 秒，需要约 300 步才能收回编译成本。

因此要把下面几项分开：

```text
编译时间
预热时间
稳态 step 时间
整个训练任务总时间
```

短任务可能不值得编译，长时间且 shape 稳定的任务更容易受益。

### 11.6 `mfu` 为什么不能直接相信

nanoGPT 的 `estimate_mfu()` 默认拿 A100 BF16 的理论峰值作分母。当前设备是 RTX 5060 Laptop GPU，因此日志中约 2%–3% 的 MFU 不能当成当前 GPU 的真实利用率。

现阶段优先报告可靠的 tokens/s、step 时间、峰值显存和 profiler 时间线。以后使用与实际硬件匹配的理论峰值和 FLOP 估算，再讨论 MFU。

---

## 12. B1：无 Profiler 基线怎样建立、为什么必要

### 12.1 B1 和 Profiler 回答不同问题

无 Profiler 基线是后续实验的对照组和统一尺子：

```text
B1 无 Profiler：当前完整训练到底有多快？
Profiler：时间具体花在哪里，下一步应优化什么？
修改代码：根据 trace 只改变一个主要变量
B1 无 Profiler重测：完整训练最终有没有真正变快？
```

Profiler 会收集事件、shape、调用栈、显存和时间线，本身会产生开销并改变运行状态。因此 Profiler 用于诊断，不能把带 Profiler 的速度直接作为正式吞吐。

B1 为后续阶段提供：

| 阶段 | 怎样使用 B1 |
|---|---|
| B2 Profiler | 先定位瓶颈，再用同一 B1 方法做无 Profiler 前后对照 |
| C DDP | 用目标服务器同机单卡吞吐计算 2/4 卡 speedup 和 efficiency |
| D MoE | 在相同窗口和工作量下比较 Dense/MoE 的速度、显存和 loss |
| E Triton | 判断单个 kernel 的微基准收益是否转化为完整训练收益 |

没有 B1 仍然可以让训练或 DDP 功能上跑起来，但不能可靠证明一次修改带来了多少端到端收益，也不能解释多卡扩展效率。

### 12.2 先区分进程、参数更新和 micro-step

当前一次完整 B1 运行的层级是：

```text
一次独立 Python 进程
├── 创建模型、优化器和 CUDA 环境
├── 20 次性能预热更新：执行，但不统计
└── 100 次正式测量更新：得到 100 个 step 时间
    └── 每次 optimizer update
        ├── micro-step 1：forward + backward
        ├── micro-step 2：forward + backward
        ├── micro-step 3：forward + backward
        ├── micro-step 4：forward + backward
        ├── gradient clipping
        ├── AdamW step：真正修改一次参数
        └── zero_grad：清空梯度，不清空参数
```

因此每个进程的正式测量包含：

```text
100 次 optimizer update
100 × 4 = 400 次 forward/backward
100 次 AdamW 参数更新
```

项目再把这整个进程独立启动三次，而不是在同一进程中接着训练三段。

### 12.3 怎样固定数据和其他变量

原版 `get_batch()` 使用有放回随机采样，适合训练，但不同运行可能抽到不同窗口。B1 使用与模型 RNG 分离的数据生成器：

```text
模型初始化 seed = 1337
数据窗口 seed = 20260920
```

测量时还把实际消费的窗口起点按顺序计算 SHA256。三次运行的窗口哈希相同，说明不是“配置看起来一样”，而是实际使用了相同的 3,200 个正式测量窗口。

同时固定：

```text
模型初始化、窗口顺序
batch=8、block_size=512、梯度累积=4
BF16、compile=False、dropout=0
模型层数、头数和宽度
```

固定 seed 不一定让所有 CUDA kernel 在所有硬件上逐 bit 完全一致。B1 不强制可能改变真实性能的全局确定性算法；它控制主要随机变量，并用 loss 有限、工作量一致和多次重复检查结果。

### 12.4 为什么要预热、测 100 次并启动 3 个进程

#### 性能预热不是学习率预热

这里的性能预热是让系统进入稳态，不记录它的时间：

- CUDA context 和 kernel 完成首次初始化；
- PyTorch CUDA allocator 建立显存池；
- GPU 从空闲频率进入稳定工作频率；
- 文件页缓存和数据路径变热；
- 优化器完成初期状态建立。

它与“训练初期逐步增加学习率”的 learning-rate warmup 不是一回事。B1 配置使用固定学习率；20 次更新只是性能预热。

20、100、3 是当前项目的最低实验协议，不是普遍不变的数学定律：

| 数量 | 作用 |
|---:|---|
| 20 次预热 | 丢弃冷启动阶段；如果 20 次后仍不稳定，应继续增加 |
| 100 次测量 | 保留足够原始值，观察典型速度、波动和离群点 |
| 3 个独立进程 | 检查重新初始化 CUDA、allocator、模型以及系统状态后是否仍可复现 |

独立进程主要检查运行间稳定性，不是用来固定数据；数据稳定已经由独立 RNG 和窗口哈希保证。

一次进程从头到尾包含：

```text
启动 Python
→ 读取配置并设置 seed
→ 读取数据元信息
→ 创建模型并搬到 GPU
→ 创建 AdamW
→ 用 memmap 取得第一批数据并 H2D
→ 20 次预热
→ 同步并重置峰值显存统计
→ 100 次正式更新
→ 同步并读取 Event、loss 和显存
→ 写 JSON
→ 进程退出并释放资源
```

### 12.5 H2D：数据怎样从 CPU 到 GPU

H2D 是 **Host to Device**：

```text
Host = CPU 和系统内存 RAM
Device = GPU 和显存 VRAM
H2D = 把 batch 从 CPU 内存复制到 GPU 显存
```

可以把 CPU 内存看作仓库、GPU 显存看作厨房操作台。当前数据路径是：

```text
磁盘 train.bin
→ np.memmap 读取窗口
→ CPU Tensor
→ pin_memory
→ .to(cuda, non_blocking=True) 做 H2D
→ GPU 上的 forward/backward
```

`non_blocking=True` 配合 pinned memory，让 CPU 尽量在 GPU 计算当前 batch 时准备和搬运下一批。如果 CPU 数据准备或 H2D 太慢，GPU 会出现等待空洞。

CUDA Event 主要记录 GPU 时间线，不会把纯 CPU 工作单独拆成一个 CPU 指标；但 CPU 太慢导致 GPU 在两个 Event 之间等待时，这种空洞会影响完整 step 时间。B2 Profiler 会进一步区分 CPU 准备、H2D 和 GPU kernel。

### 12.6 CUDA Event 与 `synchronize()` 分别做什么

CUDA 默认异步执行：CPU 提交 GPU 工作后通常立即继续，不会自动等 GPU 完成。可以把 CPU 看成下单的服务员、GPU 看成做菜的厨师：服务员把订单全部交出去，不代表菜已经做完。

两者职责不同：

```text
CUDA Event：在 GPU 自己的执行队列中放置时间戳
CUDA synchronize：让 CPU 等待 GPU 完成当前已提交的工作
```

当前每次正式更新使用一对 Event：

```text
start_event
    micro-step 1：forward / 准备下一批并 H2D / backward
    micro-step 2：forward / 准备下一批并 H2D / backward
    micro-step 3：forward / 准备下一批并 H2D / backward
    micro-step 4：forward / 准备下一批并 H2D / backward
    gradient clipping
    AdamW step
    zero_grad
end_event
```

100 次更新使用 100 对 Event，也就是 100 个 start 和 100 个 end，最后得到 100 个原始时间。

B1 在两个重要边界同步：

1. 20 次预热之后同步，确认预热任务真的完成，再进入测量区间；
2. 100 次测量之后同步，确认所有 end Event 已完成，再读取时间和 loss。

不在每个小算子后随意同步，因为频繁同步会破坏 CPU/GPU 流水和将来的计算通信重叠，人为改变被测程序。

### 12.7 为什么记录这些指标

| 指标 | 回答的问题 | 为什么不能省略 |
|---|---|---|
| 每步原始时间 | 波动、尖峰和是否随时间变化怎样？ | 只保留平均值会掩盖离群点和趋势 |
| 中位数 | 典型稳态 step 是多少？ | 比平均值更不容易被少量尖峰拉偏 |
| tokens/s | 单位时间完成多少训练工作？ | batch、累积和 GPU 数变化时只看 step 时间会误判 |
| peak allocated | 张量实际使用过多少显存？ | 用于判断 batch 余量和模型/算子额外开销 |
| peak reserved | PyTorch allocator 向 CUDA 保留多少显存？ | 更接近其他进程看到的 PyTorch 显存占用 |
| loss | 训练链路是否仍然有限且大致合理？ | 跳过 backward 也会“更快”，但不是有效优化 |
| 实际 token 数 | 对照双方是否做了同样多的工作？ | 防止通过少算数据得到虚假加速 |

当前每个更新的工作量：

```text
8 × 512 × 4 = 16,384 tokens/update
```

每个进程正式测量：

```text
100 × 16,384 = 1,638,400 个训练位置
```

吞吐计算：

```text
tokens/s = tokens/update ÷ seconds/update
```

B1 为避免每步 `.item()` 强制 CPU/GPU 同步，在测量结束后统一读取 loss。当前 JSON 记录的是每次更新最后一个 micro-batch 的 loss，用作轻量正确性哨兵，不是正式 validation loss；validation 必须放在正式计时区间之外。

### 12.8 当前 B1 的具体证据

固定配置：8 层、8 heads、宽度 512、序列长度 512、BF16、`compile=False`、micro-batch 8、梯度累积 4。

| 独立运行 | step 中位数 | tokens/s | peak allocated | peak reserved |
|---|---:|---:|---:|---:|
| run1 | 347.097 ms | 47,203 | 3,906.9 MiB | 4,910.0 MiB |
| run2 | 346.388 ms | 47,300 | 3,906.9 MiB | 4,910.0 MiB |
| run3 | 346.390 ms | 47,299 | 3,906.9 MiB | 4,910.0 MiB |

跨运行中位数为：

```text
346.390 ms/update
47,299 tokens/s
```

三次运行中位数范围只相差 0.709 ms，约为报告中位数的 0.20%；测量窗口哈希完全相同，所有记录的 loss 均为有限值。原始结果和完整边界见 [`../results/README.md`](../results/README.md) 与 `../results/b1_single_gpu/`。

这些本地结果是 RTX 5060 Laptop GPU 的 B1。C 阶段必须在目标四卡服务器的一张固定 GPU 上重复相同协议，才能用作该服务器 2/4 卡 speedup 的分母：

```text
speedup(N) = N 卡全机吞吐 / 同机 1 卡吞吐
efficiency(N) = speedup(N) / N
```

不能用本地笔记本单卡与远程服务器四卡直接计算扩展效率。

### 12.9 本轮问题形成的易错点清单

- 性能预热是执行但丢弃，不是在测“预热有多快”。
- 100 次测量是 100 对 CUDA Event、100 次 optimizer update，同时包含 400 次 forward/backward。
- 三次独立进程主要验证系统级复现性；窗口 seed 和哈希负责固定数据。
- Event 是 GPU 时间戳，`synchronize()` 是 CPU 等待 GPU 的边界动作。
- H2D 是 CPU 内存到 GPU 显存，不是 GPU 之间通信；DDP 的梯度 AllReduce 是另一条通信路径。
- step 更短不自动代表吞吐更高，必须同时核对实际 token 数。
- Profiler 用来解释瓶颈，正式优化收益仍由无 Profiler 基线裁决。

### 12.10 B2：怎样从 trace 得到优化并回到 B1 验证

B2 在相同单卡工作负载上增加了以下范围标记：

```text
get_batch
forward
backward
gradient_clipping
optimizer_step
zero_grad
```

20 次无 Profiler 更新让系统先进入稳态，随后使用：

```text
wait=2, warmup=2, active=10
```

采集 CPU/CUDA、shape 和显存 trace。eager trace 的主要证据是：

- `aten::mm` 是主要 CUDA 算子，10 个 active step 累计 2,184.5 ms self-device time；
- `aten::copy_` 有 4,880 次，累计 396.7 ms self-device time；
- `cudaLaunchKernel + cuLaunchKernel` 共 17,460 次；
- pinned H2D 总共只有 0.324 ms，不支持优先优化数据传输；
- fused AdamW 和梯度裁剪在 10 步中分别约为 44.0 ms 和 18.3 ms。

因此只改变一个变量：

```text
compile=False → compile=True
```

编译后的 trace 中，`aten::copy_` 从 4,880 次降到 160 次，合计 launch API 调用下降 20.4%，Profiler self CUDA time 下降 29.6%。这支持“图融合、减少复制/提交并选择更快计算实现”的解释；H2D 仍然很小。

最终裁决来自无 Profiler、三次独立运行的 B1 方法：

| 模式 | 跨运行 step 中位数 | tokens/s | peak allocated | peak reserved |
|---|---:|---:|---:|---:|
| eager | 346.381 ms | 47,301 | 3,906.9 MiB | 4,910.0 MiB |
| `torch.compile` | 239.897 ms | 68,296 | 3,128.6 MiB | 3,382.0 MiB |

即本机固定 shape 下得到 1.444 倍端到端加速，step 时间下降 30.74%，吞吐上升 44.39%。六次运行使用相同窗口哈希，loss 均有限且接近。

结论边界：编译时间不在稳态计时内，短任务未必能摊销；结果依赖 shape、PyTorch/CUDA 和硬件；短跑 loss 检查不等于逐 bit 等价或长期训练质量验证。Profiler 的事件层级可能重叠，因此不能把不同抽象层的百分比直接相加。

### 12.11 本轮疑问：CPU、CUDA 和 H2D 怎样连接

先建立一条数据和计算路径：

```text
磁盘上的 token
    ↓ CPU 读取、采样并组成 batch
CPU 普通内存
    ↓ pin_memory，进入适合 DMA 的锁页内存
H2D：Host to Device
    ↓ 经过 CPU/GPU 之间的互连
GPU 显存
    ↓ CUDA kernel
矩阵乘法、Attention、loss 和 backward
```

- **CPU 区域：**运行 Python 训练循环、准备 batch、调用 PyTorch，并通过 CUDA runtime 向 GPU 提交工作。
- **CUDA API/launch：**例如 `cudaLaunchKernel`，表示 CPU 在提交 GPU 任务；提交本身不等于 GPU 已经完成。
- **GPU kernel：**GPU 真正执行矩阵乘法、softmax、复制等工作。
- **H2D：**Host to Device，即把 CPU 内存中的 `X/Y` 搬到 GPU 显存。它是一段数据传输，不是另一种处理器，也不是 GPU 之间通信。

当前 `get_batch()` 先构造 CPU Tensor，再执行：

```python
x.pin_memory().to(device, non_blocking=True)
```

`pin_memory()` 让源数据位于适合异步 DMA 的锁页内存；`.to(device)` 才是 H2D。`non_blocking=True` 表示满足条件时 CPU 不必原地等待复制完成。异步不等于没有成本，仍需从 trace 判断复制是否让 GPU 等待。

本次 eager trace 的 10 个 active updates 中，pinned H2D 合计仅 0.324 ms，约 0.0324 ms/update。它远小于模型计算，因此当前证据不支持优先改 DataLoader 或 H2D。DDP 中的 AllReduce 则是 GPU/rank 之间聚合梯度，不能与 H2D 混淆。

### 12.12 本轮疑问：graph capture、graph break 和 recompile

`torch.compile` 的基本思路是先观察一段 PyTorch 运算，把可以捕获的部分整理成计算图，再对整张图做优化：

```text
逐条执行的 PyTorch 程序
        ↓ graph capture
可分析的 FX 计算图
        ↓ 融合、调度和代码生成
优化后的 GPU kernels
```

三个容易混淆的词：

| 名词 | 形象理解 | 性能影响 |
|---|---|---|
| graph capture | 把临时口头指令整理成一张施工图 | 有了整图才有跨算子优化机会 |
| graph break | 某段 Python 行为无法继续画进同一张图，先退出到 eager，之后可能再开始新图 | 图变碎，融合减少，Python/launch 开销增加 |
| recompile | shape、dtype、device 或其他 guard 条件变化，旧图不能复用，需要生成新版本 | 再次支付编译成本，频繁发生会抵消稳态收益 |

Graph break 不一定导致报错；程序通常还能以 eager 方式继续，但编译优化覆盖率会下降。Recompile 也不是错误，它是为新条件生成另一份有效实现。问题在于二者如果过多，会让编译时间增加并减少融合收益。

本项目固定 `batch=8`、`block_size=512`、BF16、模型结构和设备，shape 长期稳定，因此编译结果容易复用。换成动态 batch、动态序列长度或依赖 Tensor 数值的 Python 控制流后，必须重新观察 graph break 和 recompile，不能直接沿用本次 1.444 倍结果。

### 12.13 本轮疑问：Dynamo、AOTAutograd 和 Inductor 的分工

可以把默认编译路径记为三种角色：

```text
Python/PyTorch
      ↓
TorchDynamo：捕获员
把可编译的 Tensor 运算捕获为 FX 图，并设置复用条件 guards
      ↓
AOTAutograd：正反向设计员
整理 forward/backward 图以及反向需要保存的中间量
      ↓
TorchInductor：执行方案优化员
做融合、调度、内存规划和代码生成
      ↓
CUDA kernels 在 GPU 上执行
```

当前代码执行的是 `torch.compile(model)`，所以优化重点是模型的 forward 及其 backward。`get_batch`、gradient clipping、AdamW step 和 `zero_grad` 在这个编译边界之外。trace 中 H2D、gradient clipping 和 AdamW 前后几乎不变，而 copy、launch 和模型计算下降，与这个边界相符。

不能仅凭 key averages 断言某一个具体编译 pass 是唯一原因。当前证据足以说明编译后的模型图整体减少了复制和提交，并改善了稳态执行；若要定位到某个融合 kernel 或生成代码，还需进一步查看 Inductor 日志或专用 GPU 工具。

### 12.14 B2 的十项能力怎样分步练习

“运行两种 Profiler、打开 trace、找热点、提出假设、做 A/B、检查结果”不是十门独立技术，而是一条实验链：

```text
采集 → 阅读 → 判断 → 只改一个变量 → 无 Profiler 验证 → 限定结论
```

当前学习顺序不应是立刻重跑六次实验，而是先读取已经存在的证据：

1. 执行 `less results/b2_profiler/baseline_key_averages.txt`，找到 `aten::mm`、`aten::copy_` 和调用次数，按 `q` 退出。
2. 阅读 `compiled_key_averages.txt`，比较相同字段，不要求一开始看懂全部 CUDA kernel 名称。
3. 执行 `python -m json.tool results/b2_profiler/summary.json | less`，回答“主要热点、H2D 是否重要、compile 改变了什么”。
4. 在 Perfetto 打开 `logs/b2_profiler/baseline_trace.json`，先只找 CPU 轨道、GPU 轨道、`forward`、`backward` 和 `Memcpy HtoD (Pinned -> Device)`。
5. 用固定句式提出假设：**观察到什么 → 判断瓶颈是什么 → 只改变什么 → 预计哪些指标变化**。
6. 最后读取 `results/b2_compile_ab/summary.json`，用无 Profiler 的 step、tokens/s、显存、哈希和 loss 决定优化是否成立。

本次假设可以完整复述为：eager trace 中 copy 和 kernel launch 很多，H2D 很小；因此尝试只打开 `torch.compile`，预计 copy/launch 和 step 时间下降；compiled trace 支持机制，无 Profiler 三次 A/B 才给出正式的 1.444 倍收益。

现阶段 B2 的最低掌握标准不是能解释每个 kernel，而是能独立回答：

1. `aten::mm` 为什么是主要 GPU 热点？
2. H2D 为什么不是本次首要瓶颈？
3. 为什么 `aten::copy_` 和 launch 数量支持尝试 compile？
4. 为什么不能拿带 Profiler 的时间报告正式 speedup？
5. 为什么本次结果不能直接推广到动态 shape 和目标四卡服务器？

---

## 13. C 阶段：1/2/4 卡 DDP 怎样开展

### 13.1 第三阶段的目标和正确顺序

C 阶段不是“让四张卡都亮起来”就完成了，而是要回答五个问题：

1. 每个 rank 是否拿到了正确且可核对的数据？
2. 多卡一次更新是否与等价单卡更新基本一致？
3. 固定全局工作量时，1/2/4 卡能缩短多少 step 时间？
4. 固定每卡工作量时，总吞吐能随 GPU 数怎样增长？
5. 没有达到线性加速的时间花在计算、通信、数据等待还是最慢 rank？

正确实施顺序是：

```text
目标服务器环境与同机单卡基线
    ↓
确定性全局窗口与 rank 分配
    ↓
单卡/两卡一次更新正确性
    ↓
1/2/4 卡 Strong scaling
    ↓
1/2/4 卡 Weak scaling
    ↓
NCCL/AllReduce trace 与拓扑解释
    ↓
可恢复的完整窗口覆盖训练
```

不能先测四卡速度再补正确性。错误的数据分片也可能产生看似很高的吞吐，但它不是有效训练。

### 13.2 DDP 最小心智模型

单机四卡通常启动四个 Python 进程，每个进程负责一张 GPU，并保存完整模型副本：

| 名称 | 含义 |
|---|---|
| `RANK` | 当前进程在整个分布式任务中的全局编号 |
| `LOCAL_RANK` | 当前进程在本机上的 GPU 编号 |
| `WORLD_SIZE` | 总进程数；单机一卡、两卡、四卡时分别为 1、2、4 |
| process group | 允许所有 rank 通过 NCCL 等后端通信的进程集合 |
| DDP wrapper | 包装模型，在 backward 中触发梯度同步 |

`rank` 不是 GPU 型号、梯度等级或模型层数，而是**一次分布式任务中某个进程的编号**。单机四卡通常是 rank 0–3，各自绑定一张 GPU；rank 0 常额外负责写日志和汇总，但计算和通信上仍是参与训练的普通成员。多机时 `RANK` 仍全局唯一，而 `LOCAL_RANK` 会在每台机器上从 0 重新编号。

每个 rank 独立做 forward/backward。在 backward 过程中，DDP 对各 rank 的梯度执行 AllReduce，使所有模型副本得到一致的聚合梯度，再各自执行相同的 optimizer step。

```text
rank 0 本地梯度 ┐
rank 1 本地梯度 ├─ AllReduce 求和/平均 ─→ 每个 rank 得到一致梯度
rank 2 本地梯度 ┤
rank 3 本地梯度 ┘
```

当前 `train.py` 已有读取 `RANK/LOCAL_RANK/WORLD_SIZE`、初始化 NCCL、绑定 GPU、包装 DDP，以及只在最后一个累积 micro-step 同步梯度的基础代码。但普通随机采样还不能审计全局窗口是否重复或遗漏；B1/B2 模式也明确只允许单卡；计时与显存尚未汇总所有 rank。因此“有 DDP 代码”不等于 C 阶段已经具备可验证基准。

### 13.3 C0：目标服务器准入与同机单卡基线

先在目标四卡服务器完成只读环境记录：

```text
nvidia-smi -L
nvidia-smi topo -m
PyTorch/CUDA/NCCL 版本
GPU 型号与每卡显存
代码 commit
train.bin、val.bin、manifest 的 SHA256
```

然后固定其中一张 GPU，使用 B1 的 20 次预热、100 次测量、3 个独立进程重做单卡基线。这个结果才是服务器上 2/4 卡 speedup 的分母。

正确性阶段先使用 `compile=False` 减少变量。性能阶段在目标服务器重新确认 compile 后，只选择一个 compile 状态作为 1/2/4 卡主对照，并在所有卡数中保持一致；本地 RTX 5060 Laptop 上的 1.444 倍不能替代服务器实测。

### 13.4 C1：确定性全局窗口和正确性门槛

先生成一条与 rank 数无关的**全局窗口序列**，再按 rank 分配，不能让每个 rank 各自随意随机抽取。Strong scaling 中，每次全局更新固定 32 条长度为 512 的序列：

```text
32 × 512 = 16,384 tokens/update
```

在 `batch=8` 下：

| GPU 数 | 每 rank accumulation | 每 rank 每次更新的序列数 | 全局 tokens/update |
|---:|---:|---:|---:|
| 1 | 4 | 32 | 16,384 |
| 2 | 2 | 16 | 16,384 |
| 4 | 1 | 8 | 16,384 |

必须先在小型人工窗口列表上验证：

- 所有 rank 的并集恰好等于该次全局窗口；
- rank 之间没有重复；
- 没有窗口意外遗漏；
- 不足完整全局 batch 时明确选择丢弃还是重复补齐，并记录数量；
- 相同 seed 和更新编号能重建相同全局顺序。

随后做最小数值正确性检查：相同初始化、相同全局输入下，对比单卡大有效 batch 与两卡分片的一次更新；检查聚合 loss、梯度或更新后参数的最大绝对/相对误差，并确认两个 rank 更新后的参数彼此一致。浮点归约顺序不同可能带来小误差，因此使用明确容差，而不是盲目要求逐 bit 相同。

### 13.5 C2：新增可复现 DDP benchmark

将 B1 协议扩展到每个 rank：

- 20 次性能预热、100 次正式更新、每个卡数独立重复 3 次；
- 每个 rank 使用 CUDA Event 记录 step；
- 每次全局 step 以最慢 rank 时间作为关键时间，因为其他 rank 不能比最慢参与者更早完成有效全局更新；
- 记录每 rank 原始时间、峰值 allocated/reserved、loss 和本地窗口哈希；
- 由 rank 0 汇总全局窗口哈希、全机实际 token 数、中位数、全机 tokens/s、speedup 和 efficiency；
- 计时区间排除初始化、编译、评估、checkpoint 和 Profiler。

建议新增独立的 C 阶段配置与结果目录，避免让单卡 B1 JSON 混入多卡字段：

```text
config/benchmark_ddp_tinystories.py
results/c_ddp/strong/{1gpu,2gpu,4gpu}/
results/c_ddp/weak/{1gpu,2gpu,4gpu}/
results/c_ddp/profiler/
```

### 13.6 C3：Strong scaling

Strong scaling 固定的是**全机每次更新总工作量**：

```text
1/2/4 卡都处理 16,384 tokens/update
```

计算：

```text
speedup(N) = N卡全机 tokens/s ÷ 同机1卡 tokens/s
efficiency(N) = speedup(N) ÷ N
```

它回答：“同一份工作交给更多 GPU，能更快完成多少？”GPU 越多，每卡分到的计算越少，而 AllReduce 仍有固定代价，所以四卡通常达不到四倍。

每个卡数必须报告全机 tokens/s、step 中位数、speedup、efficiency、最慢 rank step、各 rank 显存和窗口哈希，不能只报告 rank 0 时间。

### 13.7 C4：Weak scaling

Weak scaling 固定的是**每张 GPU 的工作量**。若每卡仍使用 `batch=8, block=512, accumulation=4`：

| GPU 数 | 每卡 tokens/update | 全局 tokens/update |
|---:|---:|---:|
| 1 | 16,384 | 16,384 |
| 2 | 16,384 | 32,768 |
| 4 | 16,384 | 65,536 |

它回答：“每增加一张 GPU 并同时增加同等工作，总吞吐是否近似线性增长？”需要同时报告全机和每卡 tokens/s。Weak scaling 的全局 batch 会变大，因此它是系统吞吐实验，不应直接与固定全局 batch 的训练收敛速度混为一谈。

### 13.8 C4：通信 trace 和四卡未线性加速的解释

在 short Profiler schedule 中为每个 rank 保存不同文件名，并结合：

```text
nvidia-smi topo -m
NCCL/AllReduce 事件
backward kernel
各 rank step 时间
GPU 时间线空洞
```

依次区分：

1. 每卡计算量太小，通信占比自然升高；
2. AllReduce 本身耗时较大；
3. 梯度通信没有与 backward 有效重叠；
4. 某个 rank 数据准备或计算更慢，其他 rank 等待；
5. PCIe/NVLink/NUMA 拓扑限制了通信；
6. graph break、同步或日志让某些 rank 出现额外停顿。

Profiler 在 C 阶段仍用于解释机制，正式 1/2/4 卡吞吐继续由无 Profiler benchmark 裁决。

### 13.9 C5：可恢复的完整窗口覆盖训练

性能基准通过后，再做一次完整数据覆盖。这里必须先定义“窗口”：建议把训练 token 文件切成不重叠的长度 512 训练窗口，而不是枚举每个可能的滑动起点，否则相邻窗口会高度重复、工作量膨胀约 512 倍。

对长度为 `num_tokens` 的文件，因 `Y` 比 `X` 多向后取一个 token，可先定义：

```text
完整窗口数 = floor((num_tokens - 1) / block_size)
尾部 token 数 = (num_tokens - 1) mod block_size
```

然后确定性打乱窗口编号并按 rank 分配。若最后不足一个全局更新，应明确采用丢弃或重复补齐；“所有窗口至少一次”通常需要补齐并记录重复数。Checkpoint 除模型和优化器外，还要能恢复 epoch、全局窗口游标、seed 和补齐策略。结束时报告覆盖率、重复数、尾部数、累计 token、总时长、峰值显存以及初末 validation loss。

本机实测使用 `tail_policy=pad`：4,222 个完整训练窗口被确定性打乱，按每次更新 32 个窗口排成 132 次更新；4,224 个调度位置中有 2 个是明确记录的重复补齐，因此唯一窗口覆盖率仍为 100%。413 个不足一个 block 的训练尾部 token 没有伪装成完整窗口。完整 validation 则用变长的最后一段把全部 194,558 个 target token 纳入加权 loss，避免丢掉 validation 尾部。

Checkpoint 的意义不是“有一个 `.pt` 文件”，而是能重建下一次更新。当前保存模型、optimizer、GradScaler、每个 rank 的 RNG、`next_update`、loss 历史、配置签名和已消费全局窗口前缀哈希。自动检查把训练拆成“2 步 + 恢复 2 步”，再与连续 4 步比较；模型 SHA256、loss 历史和窗口哈希相同，才算恢复正确。

### 13.10 C6：DeepSpeed ZeRO 骨架怎样理解

ZeRO 的核心不是让 forward 变成另一种模型，而是逐阶段把训练状态分散到 data-parallel ranks：stage 1 分 optimizer state，stage 2 再分 gradient，stage 3 再分 parameter。`train_deepspeed.py` 仍使用同一个 `GPT`、窗口调度器和 AdamW 参数组，只把 backward、梯度累积边界、step 和分片 checkpoint 生命周期交给 DeepSpeed Engine。

单卡能验证 API、loss、更新边界、保存/恢复、配置解析和 checkpoint 合并，却不能证明“分片省显存”，因为 world size 为 1 时没有别的 rank 可分。当前 ZeRO-0 与原生路径在同一 4 步 smoke 中 loss 完全一致，最终参数最大绝对差 `7.451e-09`；ZeRO-1/2/3 也完成保存和新进程恢复，ZeRO-2/3 的 FP32 合并权重已由普通 `GPT` 严格加载。四卡实验的新增证据应是每卡状态减少、AllGather/ReduceScatter 事件、吞吐变化和 DDP OOM/ZeRO 可运行边界。

### 13.11 C7：FSDP2 与 DCP 骨架怎样理解

FSDP2 用 composable `fully_shard` 把模块参数表示为 DTensor。当前从每个 Transformer block 开始自底向上包装，根模型再负责 embedding、final norm 和共享 lm_head；梯度累积的非最后 micro-step 暂停同步，最后一个 micro-step 才执行跨 rank 归约。Checkpoint 使用 PyTorch Distributed Checkpoint（DCP），保存可在不同进程中重新装载的分片 model/optimizer state。

单卡的 2 步保存、恢复到 4 步已通过，并与原生/ZeRO-0 使用相同 loss 轨迹和窗口哈希。这证明接线和状态生命周期成立，不证明四卡性能优于 ZeRO-3；后者必须查看真实 ReduceScatter/AllGather、显存峰值和 step 时间。

### 13.12 当前第一要务与阶段门槛

当前本机只有一张 GPU，因此不能伪造 2/4 卡结果。本机前置项已经扩展并完成到 C5/C6/C7 的可验证部分：

1. 写出全局窗口到 rank/micro-step/batch 的确定性映射；
2. 为 1/2/4 个逻辑 rank 做纯 CPU 单元测试，验证并集、重复、遗漏和尾部；
3. 新增单卡可运行的 DDP benchmark 输出格式和汇总脚本；
4. 准备单卡与两卡一次更新正确性测试；
5. 完成 C5 全窗口覆盖、完整 validation 和确定性恢复；
6. 完成 DeepSpeed ZeRO-0/1/2/3 与 FSDP2 单卡启动、更新和恢复；
7. 到四卡服务器后依次执行 C0、两卡正确性、四卡正确性、Strong、Weak、通信 trace 和正式后端矩阵。

截至 2026-10-06，前六项已在本机完成。原始结果见 `results/c_ddp/`、`results/c5_coverage/`、`results/c6_deepspeed/`、`results/c7_fsdp2/` 和 `results/c_distributed_smoke/`。这些证据只验收单卡能回答的问题，不替代第七项的目标服务器实验。

每一步的进入门槛：

| 阶段 | 必须先通过的门槛 |
|---|---|
| 两卡启动 | 全局窗口分片的 CPU 测试无重复、无遗漏 |
| 四卡启动 | 两卡参数一致性和一次更新对照通过 |
| scaling | 1/2/4 卡工作量、哈希、loss 和最慢 rank 时间可核对 |
| 通信诊断 | 无 Profiler scaling 结果已经稳定 |
| 完整覆盖 | checkpoint 能恢复窗口进度且尾部策略已记录 |
| ZeRO 正式矩阵 | 单卡 ZeRO-0 更新对照、各 stage 保存恢复和解析配置已通过 |
| FSDP2 四卡对照 | 单卡 `fully_shard` 与 DCP 保存恢复已通过 |

---

## 14. 面对现象时的初步诊断表

| 现象 | 先确认 | 第一轮动作 |
|---|---|---|
| forward OOM | batch、序列、激活峰值 | 先减 micro-batch |
| backward OOM | 保存的激活和梯度 | 减 batch，必要时激活检查点 |
| optimizer step OOM | AdamW 状态、参数规模 | 检查状态初始化，考虑分片或减模型 |
| loss 为 NaN/Inf | 学习率、输入、精度、梯度 | 定位首次异常，检查 BF16/FP16 与学习率 |
| loss 不下降 | X/Y、数据、学习率、参数是否更新 | 先查正确性，再调超参数 |
| 显存空但 GPU 很闲 | 工作量和 CPU 数据供应 | 增 batch，并用 trace 检查空洞 |
| batch 增大但吞吐持平 | GPU 或其他资源已饱和 | 停止加 batch，开始 profile |
| 某个 step 突然慢 | eval、checkpoint、同步、编译 | 从训练计时中分离额外工作 |
| 多卡加速很小 | AllReduce、每卡工作量、最慢 rank | 看通信占比和各 rank 时间 |
| compile 首步极慢 | 编译和重新编译事件 | 分开统计编译与稳态 |

诊断原则：一次只改一个主要变量，保持数据、模型、训练工作量和计时口径可比。

---

## 15. 当前学习证据与后续深入边界

不能只因为内容在对话中出现过，就把它算作已经掌握。当前证据分为：

| 状态 | 内容 | 现有证据 |
|---|---|---|
| 已复述 | `X/Y` 错位、batch、data block、梯度累积 | 已用 `ABCD/BCDE` 和训练流程反复解释 |
| 已复述 | EOT、uint16/int64、300 步不等于完整遍历 | 已回答过对应检查问题并接受纠正 |
| 已复述 | embedding、Attention、MLP、loss、梯度、模型参数 | 已把完整前向与参数更新过程连起来；曾把“更新参数”误说成“更新 token ID”，现已纠正 |
| 已实测 | 环境、CUDA、单卡训练和 checkpoint | Shakespeare 短跑以及 checkpoint 已完成 |
| 已实测 | TinyStories 流式转换、完整文件与 SHA256 | debug/full 文件已经生成并独立校验 |
| 已实测 | 50.91M 本地候选的显存与稳定性 | batch 4/6/8 适配、300 次更新和恢复均完成 |
| 已实测 | B1 确定性采样、CUDA Event 与三次独立运行 | 三次运行窗口哈希相同，中位数 346.390 ms/update、47,299 tokens/s，运行间范围约 0.20% |
| 已实测 | B2 Profiler trace 与 `torch.compile` A/B | trace 排除 H2D 为首要瓶颈；无 Profiler 三次对照得到 1.444 倍加速，六次 loss 均有限 |
| 已实测 | C 阶段本机可完成基础 | 1/2/4 逻辑 rank 窗口测试、单进程 CUDA/NCCL 与双进程 CPU/Gloo 更新对照通过；C 单卡三次中位数为 346.479 ms/update |
| 初步理解 | OOM 处理、优化目标、吞吐与延迟区别 | 已完成解释，还需要在新现象中独立选择动作 |
| 初步理解 | H2D、CUDA 异步、Event、同步、性能预热和独立进程 | 已解释 H2D/CPU/GPU 路径并有 B1/B2 证据，还需要独立在 trace 中指出对应事件 |
| 初步理解 | graph capture、graph break、recompile 与编译流水线 | 已解释 Dynamo/AOTAutograd/Inductor 分工；还需要独立观察编译日志或 graph break |
| 初步理解 | batch 饱和、compile 摊销、Profiler、AllReduce | 已有本地 Profiler/compile 证据；还需要独立解读 trace，并在多卡阶段验证 AllReduce |

“初步理解”的内容不算失败，它表示下一步需要用测量把口头知识变成诊断能力。完成第 16 章闭卷题后，可以把能独立回答的条目升级为“已复述”。

### 后续阶段再通过代码和测量掌握

- 闭卷复述一次 B1 独立进程，以及 micro-step、optimizer update、Event 和同步之间的层级。
- 独立打开 B2 trace，指出 CPU、GPU、H2D、矩阵计算、复制和 kernel launch 证据。
- 已将单卡确定性窗口采样扩展为 DDP 全局窗口和 rank 分配；下一步在目标服务器复核 2/4 卡实际分配。
- 已完成单进程 CUDA/NCCL 与双进程 CPU/Gloo 的最小更新正确性检查；下一步做双卡和四卡 CUDA/NCCL 对照。
- 区分 strong scaling、weak scaling 和完整数据覆盖训练。
- 深入 AllReduce 重叠、NCCL 拓扑和负载不均。
- 在 MoE 阶段理解路由、expert 负载和 All-to-All。
- 在 Triton 阶段深入 LayerNorm 或实际热点的 forward/backward 内核。

---

## 16. 闭卷自测题

### A. 数据与 batch

1. 原始 TinyStories 怎样一步步变成 `train.bin`？
2. Tokenizer、batch 和 block 分别负责什么？
3. EOT 是物理文件边界吗？一个 block 能跨 EOT 吗？
4. 为什么磁盘使用 `uint16`，模型输入转换为 `int64`？
5. SHA256 能保证什么，不能保证什么？
6. 为什么随机训练 300 步不代表遍历了完整数据？
7. 给定 `X=ABCD`，写出 `Y`，并说出四个预测目标。

### B. 模型与训练

1. `block_size=512` 和 `n_layer=8` 中两个 block 有什么区别？
2. `(8, 512)` 的 token ID 经过 512 维 embedding 后是什么 shape？
3. Attention 和 MLP 分别解决什么问题？
4. 为什么 Transformer Block 输入输出 shape 相同，内容仍然发生了变化？
5. logits、loss、梯度、参数按什么顺序产生和使用？
6. 模型参数包括什么？token ID 为什么不是参数？
7. 梯度累积为什么降低峰值激活显存，却可能增加一次更新的时间？

### C. 配置与系统判断

1. 当前单卡一次 optimizer update 处理多少训练位置？
2. 4 卡时怎样用 `batch=8, block=512` 保持全局 16,384 tokens/update？
3. `n_embd` 翻倍为什么可能使许多权重接近四倍？
4. 序列长度翻倍为什么 Attention 工作可能接近四倍？
5. allocated 和 reserved 有什么区别？
6. OOM 后为什么通常先减 micro-batch，而不是先缩小模型？
7. batch 增大但 tokens/s 不再增长，下一步应该看什么？
8. 为什么 eval、checkpoint 和首次 compile 不应混入稳态吞吐？
9. 为什么 DDP 的四张卡通常达不到四倍速度？
10. 为什么不能用本地笔记本单卡和远程服务器四卡计算扩展效率？

### D. B1 性能实验

1. 为什么正式吞吐必须使用无 Profiler 基线，而不是直接报告 Profiler 运行速度？
2. 当前 `batch=8, block_size=512, accumulation=4` 为什么是 16,384 tokens/update？哪些模型参数不会出现在这个公式里？
3. 一次 optimizer update 中为什么有四次 forward/backward，却只有一次 AdamW step？
4. H2D 的 Host 和 Device 分别是什么？当前 batch 从磁盘到 GPU 经过哪些步骤？
5. 性能预热和 learning-rate warmup 有什么区别？为什么前 20 次更新不计时？
6. “测量 100 次更新”对应多少对 CUDA Event、多少次 forward/backward、多少次 AdamW step？
7. 为什么需要三个独立 Python 进程？它与固定数据窗口分别解决什么问题？
8. CUDA Event 和 `torch.cuda.synchronize()` 的职责有什么不同？
9. 为什么不能在每个小算子后都调用 `synchronize()`？
10. 原始 step 时间、中位数、tokens/s、显存、loss 和实际 token 数分别防止什么误判？
11. 当前一个 B1 进程的 100 次正式更新实际测量多少训练位置？
12. 在四卡服务器上，为什么必须先重做同机单卡 B1，才能计算 DDP 扩展效率？
13. B2 trace 为什么不支持优先优化 H2D？它支持尝试 `torch.compile` 的证据是什么？
14. 为什么 compiled trace 只能解释机制，1.444 倍正式收益必须来自无 Profiler A/B？
15. `torch.compile` 的稳态收益为什么不保证短任务、动态 shape 或另一张 GPU 也有相同结果？

### E. B2 Profiler 与编译

1. CPU、CUDA API、GPU kernel 和 H2D 在一批数据的训练路径中分别负责什么？
2. `pin_memory()`、`.to(device)` 和 `non_blocking=True` 分别起什么作用？
3. graph capture、graph break 和 recompile 各自是什么？为什么固定 shape 更利于编译复用？
4. TorchDynamo、AOTAutograd 和 TorchInductor 分别处理哪一层问题？
5. 当前 `torch.compile(model)` 为什么主要优化 forward/backward，而没有明显改变 H2D 和 AdamW？
6. `Self CUDA`、`CUDA total` 和调用次数分别能说明什么？为什么不同层级时间不能随意相加？
7. 用“观察、判断、单变量修改、预期、正式验证”五部分复述本次 compile 实验。

### F. C 阶段 DDP

1. `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE` 有什么区别？为什么通常一张 GPU 对应一个进程？
2. DDP 为什么需要 AllReduce？执行后各 rank 的模型参数为什么仍应一致？
3. Strong scaling 中，1/2/4 卡怎样保持 16,384 个全局 tokens/update？
4. Weak scaling 为什么会让全局 batch 随 GPU 数增加？它与 Strong scaling 回答的问题有什么不同？
5. 为什么必须先生成确定性全局窗口，再按 rank 分配，而不是每个 rank 独立随机采样？
6. 为什么多卡 step 应关注最慢 rank，而不能只报告 rank 0？
7. 为什么目标服务器上的同机单卡基线才是 speedup 分母？
8. 四卡达不到四倍速度时，怎样依次检查计算量、AllReduce、重叠、负载不均和拓扑？

### 一分钟口述模板

不看文档，尝试完整说出：

> 一批 TinyStories token 如何形成 X/Y，怎样经过 embedding、8 个 Transformer Blocks 和输出层得到 loss；loss 怎样通过四次梯度累积更新约 50.91M 参数；当前一次更新为什么是 16,384 个训练位置；一次 B1 进程为什么先预热、再用 100 对 CUDA Event 测量、最后同步并独立重复三次；如果出现 OOM、吞吐停滞或多卡加速不佳，分别先检查什么。

能准确完成这段口述，并能在代码中找到对应位置，就已经建立了进入 Profiler 和 DDP 阶段所需的核心心智模型。
