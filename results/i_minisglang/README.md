# I：Mini-SGLang 在线 Serving

已进入 I，当前完成 **I1 上游单卡基线复现与缺口定位**，尚未完成 I1 生命周期验收。
I2 单卡消融与 I3 在线负载继续保留待办；H2/I2 的多卡实验统一延期至后期租卡窗口。

- [I1 实测与缺口清单](i1_lifecycle/README.md)：15 请求、463 次有限 logits 检查，6 项验收缺口。
- [配置](../../config/serving/i1_minisglang.json)、[依赖锁定](../../config/serving/i1_requirements.txt)。
- [实验协议](../../docs/i1_lifecycle_protocol.md)、[进程链路与运行手册](../../docs/minisglang_service_walkthrough.md)。

先修复请求清理、结束原因、校验、worker 健康与关闭，再做正式在线性能消融。
当前结果仅说明固定版本、本机 TP1、Overlap/Graph 关闭时的行为。
