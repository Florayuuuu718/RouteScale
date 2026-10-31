# I2 同输入 attention 规划干预

延续逐层追踪协议；本轮仍只诊断，不放行全模型数值或性能。
`trace001` 已验证原输出 exact 复现、全部前缀值 exact、首个差异是 layer 0 attention。
因此固定该层 Naive warm 的同一份 BF16 Q/K/V、512 个 KV、page-size=1、GQA=2、因果语义。

分别运行 512 个 full query 和其末尾 128 个 suffix query。恢复原 Q 的 fused-QKV
stride，按原各侧页表放置相同逻辑 K/V，先要求默认 plan 的输出与原 trace exact。
再对两侧共同施加 `disable_split_kv=True` 和固定 `fixed_split_size=128` 两种单项干预。
记录 FA2 原始 plan vector、CTA tile、split-KV 标志和实际 KV chunk size。

使用独立连续 FP32 SDPA math 及 FP64 math 检查绝对位置的 bottom-right causal mask；
suffix 的 query 位置是 384..511，不能使用默认从 0 开始的三角 mask。
报告每种干预的 full/suffix 差异、相对高精度参考误差和是否复现原 trace。
不根据哪一项通过而覆盖 v1 门槛。若规划干预能消除局部差异，还必须另做全模型/全部
配置的独立数值验证，且不能把原 HF FP32 等价失败改成通过。
