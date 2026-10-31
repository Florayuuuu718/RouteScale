# I2 本机 FlashInfer 形状控制范围：验收通过

2026-10-31，RTX5060 Laptop 8 GiB、Qwen3-0.6B BF16、TP1、共享tokenizer、FlashInfer fa2。
**七配置、三独立进程、21/21正式进程及160项性能审计检查通过。**
前置资格验证12544个全词表向量全部逐元素exact；同策略HTTP复验84请求、280项检查通过。
新增12项CPU检查通过。所有专用进程正常退出，未使用付费资源。

本范围明确应用 **prefill no-split + linear tile512**。原HF FP32全模型门槛仍失败，
原v1正式性能仍为0/21；这里通过的是同一BF16引擎内部策略等价及该控制策略下的系统消融。
不能将这些数据称为未改变计算形状的上游原生性能、在线SLO或自然生成质量结论。

## 结果与不确定性

下表为 **batch=3、shared prefix、warm**。吞吐与首结果均为三个独立进程均值的中位数；
每进程五次计入统计的重复，另有三次预热。首结果是 Scheduler first-result，单位ms，排除HTTP。
linear行数指每层每个受控projection的补齐行数之和，不是FLOPs。

| 配置 | 输出token/s | 首结果ms | prefill query | linear补齐行数 | 进程间CV |
|---|---:|---:|---:|---:|---:|
| Naive | 121.63 | 66.51 | 1536 | 1536 | 3.25% |
| Radix | 115.73 | 50.29 | 384 | 512 | 7.43% |
| chunk256 | 117.40 | 63.01 | 384 | 1024 | 21.43% |
| chunk128 | 112.17 | 90.62 | 384 | 1536 | 3.69% |
| Graph `[1,2,4]` | 376.60 | 50.27 | 384 | 512 | 4.02% |
| Graph `[1,3]` | 384.96 | 47.16 | 384 | 512 | 2.48% |
| Overlap | 125.81 | 59.54 | 384 | 512 | 18.69% |

- Graph相对Radix的本case配对吞吐倍率：`[1,2,4]`中位3.37×、三轮范围2.90–3.51×；
  `[1,3]`中位3.42×、范围3.03–3.59×。本范围支持Graph改善吞吐；不据小样本宣布通用最优bucket。
- Prefix复用减少了真实query。batch3 shared warm中，query从1536降至384，首结果中位数
  从66.51降至50.29ms，但配对吞吐倍率范围0.93–1.02×，没有稳定的吞吐优势。
- chunk的真实行数和补齐成本必须一起看。上述warm query均为384，chunk256/128却分别
  计算1024/1536个补齐行；本固定cohort没有普遍的chunk吞吐收益。在线长短请求的公平性留待I3验证。
- Overlap在上述case的配对倍率为1.15×，范围1.09–1.37×；batch1 shared warm范围
  为0.86–1.13×。收益依赖负载，首结果也可能增加。

全部56个统计cell的进程间CV为0.59%–21.43%。部分小收益和排序不稳定，不能只挑最佳值。
图中的点为独立进程均值，误差线是观察到的min/max，**不是置信区间**。
1344次波结束后的遥测温度47–66°C；这些快照不测kernel执行期，不能证明锁频或唯一归因噪声。

## 完整数据与图

- [独立性能审计及全部检查](perf001/audit.json)、[21进程账本](perf001/processes.json)。
- [全部56个统计cell CSV](perf001/statistics.csv)、[全部48组配对比较CSV](perf001/comparisons.csv)。
- [吞吐PNG](perf001/throughput.png) / [SVG](perf001/throughput.svg)。
- [Scheduler首结果PNG](perf001/first_result_ms.png) / [SVG](perf001/first_result_ms.svg)。
- [图表/CSV哈希、遥测范围与导出版本](perf001/export.json)。
- [新输入数值与HTTP资格](../i2_shape_qualification_v1/README.md)、
  [两个首分叉的因果证据](../i2_numerical_trace_v1/README.md)、
  [原HF FP32失败](../i2_single_gpu_v1/README.md)。
- [替代后端有界探针](../i2_backend_capability_v1/README.md)：当前固定依赖中的FA4接口错误，
  TRTLLM返回Unsupported architecture；没有fallback或替代后端性能排名。

## 范围与复现

输入为资格验证的control512子集：batch1/3、512输入、64固定续写、prefix0/384、cold/warm。
cold表示prefix条目起始为空；chunk调度可能在同一cohort内产生复用，实际query量已记录。
所有配置使用相同2048物理页、1024窗口、4执行槽与BF16数学设置。三个轮次按正序、逆序、
轮转运行，同机GPU锁串行执行。参数、模型、补丁树、工作负载和计算源码均冻结并哈希核对。

Python adapter检查、linear尾部补零/分块及其额外运算都计入wall时间。没有逐步logits CPU复制，
batch轨迹计时后落盘；终态、GPU完成、页/table唯一性、实际Graph回放和padding均独立核对。
完整中间张量与mutable日志在忽略的runs目录，Git保留源码、快照、小型数据和失败。

```bash
# 已有本机资格数据的复核
runs/envs/mini-sglang/bin/python scripts/recheck_i2_shape.py --kind findings
runs/envs/mini-sglang/bin/python scripts/recheck_i2_shape.py --kind performance \
  --directory results/i_minisglang/i2_shape_performance_v1/perf001
```

审计生成脚本拒绝覆盖已有审计；`recheck_i2_shape.py`只读复算并核对已有JSON。
新实验使用新attempt。干净环境先按数值/HTTP协议重新生成资格记录，并在
`config/serving/i2_shape_performance.json`将两条资格路径指向本次新目录，再执行：

```bash
runs/envs/mini-sglang/bin/python scripts/run_i2_shape_performance.py --attempt fresh_perf001
runs/envs/mini-sglang/bin/python scripts/audit_i2_shape_performance.py \
  --directory results/i_minisglang/i2_shape_performance_v1/fresh_perf001
runs/envs/i2-report/bin/python scripts/plot_i2_results.py \
  --directory results/i_minisglang/i2_shape_performance_v1/fresh_perf001
runs/envs/i2-report/bin/python scripts/export_i2_shape_report.py \
  --directory results/i_minisglang/i2_shape_performance_v1/fresh_perf001
```

报告环境单独创建：`uv venv --python 3.11 runs/envs/i2-report`，再用
`uv pip install --python runs/envs/i2-report/bin/python -r config/serving/i2_report_requirements.txt`。
它不修改Torch/CUDA服务环境。

I3在线负载、H3计时复验、完整交付和TP2/4仍按后续计划推进；先完成本机再集中租卡。
本轮只整理工程协议/结果，个人复盘、面试指南和知识验收仍留待实验完成后统一写。
