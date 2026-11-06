# 集中租卡选型与开机准备

核价日期：2026-11-06。用户已明确没有固定预算上限，选择标准是完成当前实验的
总费用尽可能低。本文件是选型与执行安排；实际节点尚未租用，四卡库存及成交价尚未确认。

## 推荐组合与价格依据

优先检查 **算家云青春版，同机 4×RTX 4090D 24GB，按量计费**。
若没有合格节点，再看同平台 4×4090 24GB；AutoDL 同机四卡为备选。
不能将四个单卡实例拼成当前实验要求的同机四卡。

| 候选 | 官网每卡时价 | 四卡每小时算术值 | 4 小时算力费 | 8 小时算力费 |
|---|---:|---:|---:|---:|
| 算家云青春版 4090D 24GB | ¥1.14 | ¥4.56 | ¥18.24 | ¥36.48 |
| 算家云青春版 4090 24GB | ¥1.24 | ¥4.96 | ¥19.84 | ¥39.68 |
| AutoDL 4090 24GB | ¥1.88 | ¥7.52 | ¥30.08 | ¥60.16 |
| AutoDL 3090 24GB（低价备选） | ¥1.32 | ¥5.28 | ¥21.12 | ¥42.24 |

价格来自[算家云首页](https://suanjiayun.com/)及
[AutoDL 首页](https://www.autodl.com/home)。首页价格不是四卡订单报价，
促销、会员、区域、库存和附带资源均以下单页为准；表中不再次叠加会员折扣。
这组时间只是费用敏感性计算，**不是实测完工时间或最低总价承诺**。
算家云早期 1.24 元活动公告已过活动期，本表依据当前首页展示，不能凭旧公告下单。

当前主目标是低价 4090D；4090 的时价高约 8.8%，只有完成整轮更快才能抵消差价，
暂没有这两个租用节点的实测数据。只为比较卡型而租两台机器会增加部署与重复资格费用。
3090 保留为有合适四卡节点时的候选，不把更低单卡时价自动等同于更低完工费用。

当前冻结模型为 Qwen3-0.6B 和 57,461,760 参数的小 GPT。
本机完整 F 资格的峰值 allocated 为 4,580,520,960 bytes，详见
[实际记录](../results/f_ep/local_v1/training/cuda1_full/rank0.json)；这不是目标机总显存上界。
24GB 提供了足够的选型余量，目标机仍须通过原显存和数值资格。
本轮不为 48/80GB 显存另付溢价，C1–C7 也不重复跑。

## 创建实例前的筛选条件

- 一个 Linux x86_64 实例可分配四张同型号、同显存的完整 GPU；当前实验期间独占这四张卡。
  不能用共享 vGPU、不同型号拼卡或跨主机实例替代。
- 宿主 NVIDIA 驱动建议 **580.65.06 或更新**，适配现有 CUDA 13.0 JIT 工具链及
  三套 cu129 Torch。依据 [NVIDIA CUDA 13.0 发布说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
  与[兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。
  镜像名写 CUDA 12/13 不能代替宿主驱动版本；不在容器中尝试升级宿主驱动。
- 基础镜像优先 Ubuntu 22.04/24.04；有 Git、Python3、uv、C/C++ 编译器、开发头文件、
  SSH、tar、gzip。Python3.11.16 和三套依赖由现有安装器重建，不依赖镜像预装 Torch。
- 建议至少 16 vCPU、64GB 主存，若四卡套餐自带 128GB 或更多则直接使用，
  不为超出当前需求的 CPU/内存单独加价。这是部署规划值，目标资格检查仍决定能否执行。
- 数据盘建议 200GB；项目、虚拟环境、uv/Python 缓存、模型、工具链和结果都放数据盘。
  本机包约 21.2GiB，恢复加三环境目录约 41GiB，另需外部依赖缓存、原始张量和 trace 空间。
  50GB 默认数据盘不够保留完整部署和原始证据。
- 核对外网依赖下载速度及 SSH 上传能力。现有包带五个大型 H1 wheels，其他锁定依赖
  仍需网络或完整 uv 缓存；不得将其标成完全离线镜像。
- 先核对是否处于资源下架通知期。青春版不支持无卡开机及升降配置，见
  [官方版本对照](https://suanjiayun.com/sj-major)。不得套用 AutoDL 无卡方案。

4090 没有 NVLink，见 [NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)。
实际 PCIe/P2P/共享内存通信路径以目标拓扑和 NCCL 测试为准，不能假定消费卡四卡直连。
当前 TP 小模型的多卡性能可能低于 TP1，这仍是有效实验结果；不为获得加速比而换验收门槛。

## 费用控制与具体执行

1. 先在控制台核对候选节点的四卡总价、驱动、CPU/内存、磁盘价格和可用期限。
   优先满足以上条件的低价 4090D。若低价只支持单卡或驱动不合适，跳过该节点。
2. 所有本地准备与打包完成后再开机。青春版从开机起计费，连续完成传输、安装、
   资格、性能、trace、原机汇总和回传，减少空转；按量按秒结算，不预购全天/包周。
3. 在同一个实例顺序跑完整冻结矩阵：公共三栈/NCCL1/2/4 → H2/I2 TP 全资格 →
   F 模块与完整训练资格 → 无 Profiler 性能 → 独立 trace。正式 TP1 基线也在该机重跑。
   详见[执行手册](rental_execution.md)。不能用本机单卡成绩拼接租用机 TP2/4 加速比。
4. 控制器仍保留有限超时防止挂起。建议首会话 `max_runtime_s=43200`（12 小时），
   `max_cost=实际四卡每小时价×12`；这只是实验进程的故障保护值，不是用户预算上限、
   完工预测或一次购买 12 小时。安装、上传、导出和磁盘另计，不受控制器费用字段约束。
   只在真实节点确定后填 `existing_host_confirmed=true`、hostname 和四个 GPU UUID。
5. 资格通过后继续完整矩阵；故障立即停止新增 GPU 工作、保留现场并及时关闭云实例，
   不反复付费盲重试。正常运行中按真实日志更新剩余时间估算。控制器不会自动停止云计费。
   成功阶段的 resume 保留原截止时间，失败或半轮不能自动 resume。
6. 原机完成汇总，回传原始 session、报告和部署 manifest，核对校验和。
   关闭实例后检查控制台计费状态；确认数据已回传后释放实例/付费磁盘，避免存储续费。

算家云[官方计费规则](https://suanjiayun.com/en/help/68b6bc64482ba172c827c2df)
列出数据盘免费 50GB、超出部分 ¥0.01/GB/日；若选择总计 200GB，则额外 150GB
约 ¥1.50/日，按当日最大容量计费，关机仍收。跨两天会有两天磁盘费，以订单为准。
算力费用为 `四卡实际时价×全部开机小时数`，再加磁盘及其他明确列示费用。

若改用 AutoDL，可用无卡模式先传输并安装已有 wheels，
[官方说明](https://www.autodl.com/docs/save_money/)为 0.5 核、2GB、¥0.1/小时。
内存小，串行安装；需要 GPU 的导入/JIT 与资格留到四卡开机。无卡模式不预留四卡，
切回前须确认同机资源仍可用。它能降低长时间上传的成本，是否比青春版更省取决于耗时：
以表中 4090D/4090 时价，忽略磁盘差额，青春版成本约 `4.56×(计算小时+准备小时)`，
AutoDL 约 `7.52×计算小时+0.1×准备小时`，还需加四卡模式中的安装/JIT时间。
若准备很慢，不能仅凭 GPU 标价选青春版。

## 本机传输材料与上机命令

已验收包：`runs/rental/bundle002`，4967 个资产/源码文件，22,812,790,808 bytes。
manifest SHA256：`efce3d8658750f3677d3613c003f3b151342c62086cdbad03cc76589b6e82b12`。
它固定源码 `7b4018a3339d09122987136d9146e537a408ec2d`；后续文档和结果提交不改变该包。

传输压缩包为 `runs/rental/bundle002-compact.tar.gz`；旁边的 `.sha256` 用于整包校验，
`transport002.json` 记录压缩后大小、实际解压后的逐文件校验和原 manifest 的对应关系。
原包约 12.4GB 内容重复，传输 tar 用标准硬链接引用相同内容，目标解压还原全部路径。
压缩仅改变传输格式，不改变实验源码或验收资产；包及原始资产都不进入 Git。
压缩后为 7,604,609,704 bytes（约 7.1GiB），较原逻辑大小减少约 66.7%；
已实际解压到新目录，4967 个文件及原 manifest 全部通过校验，见
[传输包验收记录](../results/rental/local_v1/transport002.json)。

用 SSH 配置中的主机别名 `routescale-rental` 代替实际地址，密钥留在本机 SSH 配置。
以下 `/DATA/routescalelab` 必须替换为该实例**实际数据盘**上的新目录；AutoDL 通常使用
`/root/autodl-tmp/routescalelab`，其他平台先看挂载，不能假定 `/workspace` 是数据盘。

```bash
rsync -avP runs/rental/bundle002-compact.tar.gz runs/rental/bundle002-compact.tar.gz.sha256 \
  routescale-rental:/DATA/routescalelab/
```

在目标数据盘目录执行：

```bash
sha256sum -c bundle002-compact.tar.gz.sha256
tar -xzf bundle002-compact.tar.gz
git clone bundle002/project.bundle RouteScale-bootstrap
python3 RouteScale-bootstrap/scripts/prepare_rental.py restore \
  --bundle bundle002 --root RouteScale
export UV_CACHE_DIR="$PWD/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/uv-python"
cd RouteScale
python3 scripts/prepare_rental.py install
python3 scripts/prepare_rental.py check-env
```

`restore` 自动复核原包逐文件 SHA。源码目录必须保持干净，凭据和主机配置留忽略路径。
三环境安装后使用[执行手册中的正式入口](rental_execution.md#执行停止与继续)，
现场主机配置未填好时不执行 GPU 矩阵。汇总须在原主机完成，不能先销毁主机再尝试本地审计。
