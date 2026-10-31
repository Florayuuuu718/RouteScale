# I2 外部 HF 参考逐层对齐协议 v1

本轮先于 I3，定位原 BF16 target 对 HF FP32 的失败。原配置、门槛、失败证据不改。
不计时、不租卡；所有输出均 `performance_allowed=false`。诊断进程完成与数值通过
分别记录，负对照的预期数值失败不应伪装为进程异常。

## 冻结输入与参考

使用原 I2 v1 workload、BF16 checkpoint 和独立 HF eager FP32 `attempt001/reference.pt`，
绑定文件、模型、代码、上游树和环境哈希。原门槛保持 `atol=.125, rtol=.02,
relative_RMSE<=.005`，全部步骤、全部151936词表元素有限且通过。TF32、BF16/FP16
reduced-precision reduction 全部关闭。先验证 checkpoint 全部张量为 BF16，避免
FP16 模式改变权重取值来源。

pilot：四种 case 保留原 cold 批次、各四步，共32向量。仅首请求首个512-token prefill
记录末128行全部28层算子，GPU clone 后整请求复制 CPU。HF hook 不改变返回值；
Mini 复用原只读追踪器。pilot HF FP32 和原生 BF16 输出必须复现各自旧证据。
full：原四 case、batch1/3、zero/shared、cold/warm、全部64步，共1024向量。
pilot 只用于定位，不代表完整门槛通过。

## 明示干预

- `hf_fp32`：同一 BF16 权重升 FP32 运算的原参考实现。
- `hf_bf16`：HF eager BF16 精度负对照。
- `native_bf16`：原 Mini BF16，作为只读观察器复现对照。
- `residual_fp32`：Mini BF16 主算子不变，残差保留 FP32，归一化计算 FP32 后转 BF16。
  这是残差与 norm 的联合干预，不能唯一归因给单个 rounding 操作。
- `native_fp16`：实际 Mini/FlashInfer 路径改用 FP16 激活、权重和 KV；相同 BF16 权重
  转 FP16。独立报告有限性与门槛，绝不回填 BF16 通过，也不作为 BF16 性能。
- `fp32_math`：Mini 模型/合并投影/真实 scheduler/页表保留；同一 BF16 权重升 FP32，
  norm、RoPE、激活和真实物理页上的 attention 显式使用 PyTorch FP32 math。
  此模式绕过 FlashInfer attention 执行，只验证所覆盖的模型和调度/缓存语义，不能
  证明原 FlashInfer 算子全部正确。只允许 Graph/Overlap 关闭。

首个外部差异按算子实际数据流报告；HF split projection 合并后与 Mini merged
projection 对齐，norm 输入残差单独报告。不同精度下的首个非 exact 不自动等于 bug。
使用相同算子输入的局部舍入探针解释误差。通过需独立重开张量复算，不依赖 worker
自报；全模型语义和算子局部分析各有边界。

每一轮使用新目录并冻结源码；后续修正不能继续旧轮。新精度策略即便通过原阈值，
也必须写出目标精度改变与成本未知，原 BF16 HF 等价性仍按原结果保留。
