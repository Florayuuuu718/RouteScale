# I2 首个分叉层诊断 v1

本轮是数值诊断，不签发性能 gate，不改 I2 v1 的参考、容差、源码树或失败记录。
使用原 `b1_shared` 的 512-token cold/warm 输入和全部 64 个固定续写 token。
Naive 与 Radix 分别在独立进程执行，仍为原生 BF16、FlashInfer fa2、Graph/Overlap 关闭。

只观察 cold/warm 的第一次 prefill。cold 保存各层前 384 个逻辑 K/V；warm 保存每层
最后 128 个实际计算位置的 norm、projection、Q/K/V、attention、MLP 输出，以及完整
512 个逻辑 K/V、Radix 写入前的 384 个缓存 K/V、物理页索引、写地址、位置与调用 metadata。
所有 GPU 数据先 clone/gather，整波完成后才搬到 CPU，不替换模型算子或修改返回值。

独立审计要求：

1. 两次新运行的 cold/warm 全部 64 步 logits 必须与原未追踪运行逐元素 exact，才能用
   追踪定位原分叉；否则标为追踪干扰，不能据此归因。
2. 输入 token、warm prefix 命中量、position、query/KV 长度、唯一页索引、写地址和
   资源释放符合原协议；prefix 数据在 cold、warm 写入前后和 Naive 重算之间逐层核对。
3. 按实际计算顺序找首个不同输出，报告 exact、最大绝对误差、相对 L2、不同元素个数。
   不能只看模型最终 top-1。
4. 若首个差异在 attention，下一项隔离实验在同一份冻结 Q/K/V 上比较 full/suffix
   query 调用与 FP32 因果数学参考；若在其他算子，则对该算子做相同输入干预。

即使局部误差很小，也不等于原全模型门槛已通过。不能放宽 v1 的门槛后覆盖旧失败，
不能将 clone/gather 的诊断时间纳入正式性能统计。原始张量放入忽略的 `runs/validation`，
小型误差、SHA256、source snapshot、原运行关联和 GPU 退出记录进 Git。
