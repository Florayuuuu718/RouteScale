# I2 后端 API 有界探针

本机 sm120，固定 Torch2.9.1+cu129、FlashInfer0.7.2.post1、sgl-kernel0.3.21。
**两项替代后端在当前固定依赖栈下均未取得内核输出，不具备本轮比较资格。**
没有使用 fallback，没有吞吐排名，也没有改动已验收环境来追求后端可用。

| 显式接口 | 实际结果 | 证据 |
|---|---|---|
| Mini-SGLang 对应 FA4 调用 | 导入 `flash_attn_origin.cute` 时缺少 `cutlass._mlir.dialects.nvvm.RoundingModeKind`，接口未能调用 | [fa4.json](cap001/fa4.json) |
| FlashInfer TRTLLM context | 编译固定版本 JIT module 后，`TllmGenFmhaRunner` 返回 `Unsupported architecture` | [trtllm.json](cap001/trtllm.json) |

这是明确接口、固定依赖和给定形状的可用性证据，不是所有软件版本/所有形状的硬件能力结论。
探针为 query32、KV128、16/8 heads、head_dim128、BF16、page64；它不构成全模型数值验收。
本轮正式消融始终是显式 FlashInfer fa2、page1，没有把这些错误转成其他后端结果。

每接口独立进程、180秒预算；FA4约14秒结束，TRTLLM约167秒结束，均正常退出且GPU回到0 MiB。
TRTLLM 在本地忽略的缓存中获取 NVIDIA 公开 JIT 元数据并编译，不涉及租卡或付费资源。
后续若升级 CUTLASS/FA4 等依赖，应放入新环境，重新做完整数值和系统验收。

```bash
runs/envs/mini-sglang/bin/python scripts/probe_i2_backend_capability.py --attempt fresh_cap001
```
