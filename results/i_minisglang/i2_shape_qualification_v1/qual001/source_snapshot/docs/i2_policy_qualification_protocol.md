# I2 形状控制策略资格验证 v1

本协议冻结在以下七配置、新输入资格验证之前。它只回答同一原生 BF16 引擎中，
缓存、chunk、Graph、Overlap 策略是否保持相同逐步模型结果，不声称通过原 HF FP32
全模型误差门槛。原 v1 失败保持不变；本资格验证本身不授权性能计时。

原 b1_shared 已逐层定位两个相继的首个分叉：FlashInfer full/suffix split-KV 自动
规划变化，以及相同输入的 BF16 MLP down GEMM 行形状变化。两侧共同施加 prefill
no-split 与 linear tile512 后，原 cold/warm 128 个全词表向量 exact。

本轮所有配置共同施加这两项显式 adapter；原生源码树保持不变。仍使用 BF16 权重、
BF16 激活、同一 sampler/固定续写、2048 物理页、page1、窗口1024、4 执行槽、TP1。
不修改 Decode 或 LM head，不掩盖 Graph padding 或真实异步 Overlap。
tile512 确实增加尾部填充运算，任何后续性能必须命名为此控制策略的性能。

输入集合：

- control512：原 seed20261031、batch1/3、prompt512、prefix0/384、输出64，16 请求。
- holdout768：新 seed20261101、batch1/2、prompt768、prefix0/512、输出64，12 请求。
  该集合未参与首层定位或策略选择。相同 builder/语料，使用独立 request ID 命名空间。

先在独立进程执行受控 Naive 参考，再在七个新进程分别重放 Naive、Radix、chunk256/128、
Graph[1,2,4]/[1,3]、Overlap。参考与目标均应用共同策略，全部28请求、1792个全词表
向量逐步对照；七配置共12544个目标向量。容差保持 atol=.125, rtol=.02, 相对RMSE<=.005，
并报告字节/逐元素 exact。冷KV填NaN、实际命中、输出预算、table/页唯一性和释放均检查。

各进程仍执行完整输入集合以保留覆盖与失败细节，出现数值失败后不运行性能。
冻结源码/模型/工作负载和真实 Graph capture/replay 记录。独立审计重开完整张量；
若有新分叉则保存该轮失败、回到对应算子定位，不改阈值、不直接补写 passed。
只有全部配置/新输入验证及受影响 HTTP 生命周期重验完成，才可另立性能范围。
