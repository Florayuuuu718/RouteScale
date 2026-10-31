# I2 完整模型的共同 no-split 规划干预

同输入 attention 探针已精确复现原差异。默认 full query 的 split-KV=false，suffix
query 的 split-KV=true、实际 chunk=171；两侧共同 no-split 或固定 chunk128 时，
同输入局部输出 exact。因此先用完整模型检查共同 no-split 干预。

这不是 HF FP32 门槛通过，也不是正式性能范围。原 I2 v1 保留失败。
不改原生源码树、不改变 BF16 权重/激活、工作负载、采样控制、资源池、容差或生命周期。
仅给每个独立进程的 FlashInfer prefill wrapper.plan 显式传入 disable_split_kv=True，
两侧同时应用；Decode 不受此 adapter 修改。

先重放原 b1_shared cold/warm 的 64 步，比较 Naive/Radix 全词表输出，核对真实命中与
页/table 释放；记录 Naive 是否仍与原未干预 Naive exact。若仍分叉，保留失败再定位，
不能据此启动性能。若通过，再另立带未参与定位输入的资格验证，覆盖全部七配置；
仍需明确区分引擎内部策略不变性与跨 HF FP32 的全模型数值误差。
