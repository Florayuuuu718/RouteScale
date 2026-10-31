# I2 精度验收：重试目录与独占 GPU

本说明补充已冻结的 [精度协议](i2_precision_protocol.md)，不修改其计算策略、
输入或数值阈值。数值资格验证后，服务关闭修复和重试编排单独归档。

`config/serving/i2_precision.json` 中 `lifecycle_directory=http001` 是首次尝试的
默认位置。该尝试失败后，不能覆盖失败目录，也不能为了改位置而修改数值验收
所绑定的配置。后续使用 `run_i2_precision_followup.py`，在 provenance 的
`artifact_bindings` 中显式选择实际数值与 HTTP 证据。正式计时还绑定独立
shutdown 审计的哈希。计算 worker 和配置保持原样。

已保留的重试：

| 尝试 | 结果与处理 |
| --- | --- |
| http001 | 首配置正常请求/取消通过，解释器回收阶段超过原 4 秒 SIGINT 宽限而升级 SIGTERM；失败保留。 |
| http002 | 无升级退出，但策略安装覆盖了 `workers_stopped` 只读观察钩子；仍按原标准失败。随后将核心策略安装放到观察器安装之前。 |
| http003 | 14 个配置的 168 请求、560 项功能检查全部通过，但从第 2 个配置起，Windows 上无关的 Anaconda Python 占用 92 MiB；独立进程日志审计拒绝该轮。 |

服务关闭策略在所有 worker 之间共享 6 秒 SIGINT、1.5 秒 SIGTERM、0.5 秒
SIGKILL 的预算，总计 8 秒；外部 10 秒门槛和健康退出不得升级的要求不变。
`serving/i2_worker_shutdown.py` 是明确的服务策略，不修改 vendor 或观察器的
判定规则。真实双 CPU 进程忽略 INT/TERM 的故障探针在 7.505 秒内被清理，
证明升级预算有界；这不代替真实模型服务的健康关闭验收。

新编排在启动前要求 GPU 显存回到 `0 MiB`，每个独立进程之后也检查原始
`gpu_after`，若再次出现占用则立即停止并保留该轮。WSL 的 `nvidia-smi` 可能
看不到 Windows 计算进程；遇到不明占用时，可用 Windows `nvidia-smi.exe`
只读核对。不得未经授权停止其他实验。

重验必须使用新目录。例如在 GPU 独占条件恢复后：

```bash
runs/envs/mini-sglang/bin/python scripts/run_i2_precision_followup.py \
  --stage http --attempt http004
runs/envs/mini-sglang/bin/python scripts/audit_i2_precision.py \
  --stage http --directory results/i_minisglang/i2_precision_v1/http004
runs/envs/mini-sglang/bin/python scripts/audit_i2_precision_shutdown.py \
  --directory results/i_minisglang/i2_precision_v1/http004
```

三个独立门槛通过后，使用所选 HTTP 目录进行 42 进程计时。正式计时期间不
并行运行测试、审计或其他 GPU 任务。数值/HTTP 审计器的原 SHA 保持不变；
最终审计器通过显式绑定重新计算这些门槛。

```bash
runs/envs/mini-sglang/bin/python scripts/run_i2_precision_followup.py \
  --stage performance --attempt perf001 \
  --lifecycle-directory results/i_minisglang/i2_precision_v1/http004
runs/envs/mini-sglang/bin/python scripts/audit_i2_precision_final.py \
  --directory results/i_minisglang/i2_precision_v1/perf001
runs/envs/i2-report/bin/python scripts/export_i2_precision.py \
  --directory results/i_minisglang/i2_precision_v1/perf001
runs/envs/mini-sglang/bin/python scripts/accept_i2_precision.py \
  --performance results/i_minisglang/i2_precision_v1/perf001
```

若示例中的目录已存在，应选择新名字；`artifact_bindings` 才是最终审计的实际
位置依据。旧控制器/审计器的 performance 默认入口仍绑定失败的 http001，
不能用于后续重试。原数值协议的冻结快照和失败门槛均保留。

本轮 CPU 记录分开保存：精度/原始准入 9 项、关闭预算 2 项、实际重试准入
9 项，共 20 项。最后一组覆盖显式位置、三个失败门槛、模型变更、HTTP 审计
变更、shutdown 审计器变更和 GPU 占用，验证当前实际计时入口。
