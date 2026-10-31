# I2 共同 no-split 与 linear tile512 干预

`trace_nosplit001` 精确复现无追踪 no-split 的 64 步输出，并确认首个分叉已从
layer0 attention 移至 layer0 MLP down_proj。此前的全部算子输出 exact；down_proj
输出有 251 个元素不同，最大绝对差 0.001953125、相对 L2 约 0.00010721。

下一项完整模型干预同时在两侧保留 no-split，并将每层 QKV/o/gate-up/down 的 BF16
prefill linear 切成固定 512 行，尾部补零后只返回真实行。不改 embedding、norm、
RoPE、attention 语义、LM head、Decode、权重/激活精度或 token 输入。
所有变体必须共同应用，明确承担额外填充运算；不能称作未修改原生计算形状的性能。

先验证原 b1_shared 的全部 128 个 cold/warm 全词表向量，再用新 seed/长度资格验证。
仍使用原逐元素容差和 relative_RMSE<=.005，额外报告 exact。每个干预使用新目录，
原 no-split 失败和 HF FP32 失败保持不变。本轮没有性能授权。
