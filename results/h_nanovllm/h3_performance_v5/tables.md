# H3 同模型性能结果

三次独立进程的中位数；固定 token 回放，全部 BF16，Prefix off。

| 范围 | case | 路径 | 输出 token/s | TTFT ms | TPOT ms | 峰值 allocated MiB |
|---|---|---|---:|---:|---:|---:|
| model_execution | batch_short | native_kv | 1169.7 | 7.152 | 6.768 | 182.5 |
| replay_e2e | batch_short | native_kv | 990.9 | 8.172 | 8.003 | 182.5 |
| model_execution | batch_short | paged_eager | 1059.3 | 7.196 | 7.499 | 185.5 |
| replay_e2e | batch_short | paged_eager | 1046.1 | 12.142 | 7.564 | 185.5 |
| model_execution | batch_short | paged_graph | 6712.1 | 12.675 | 1.038 | 193.7 |
| replay_e2e | batch_short | paged_graph | 3862.9 | 14.444 | 1.920 | 193.7 |
| model_execution | latency_short | native_kv | 159.1 | 7.221 | 6.278 | 115.0 |
| replay_e2e | latency_short | native_kv | 155.1 | 7.748 | 6.322 | 115.0 |
| model_execution | latency_short | paged_eager | 149.4 | 7.846 | 6.727 | 171.1 |
| replay_e2e | latency_short | paged_eager | 143.7 | 9.860 | 6.950 | 171.1 |
| model_execution | latency_short | paged_graph | 809.0 | 12.688 | 0.899 | 179.2 |
| replay_e2e | latency_short | paged_graph | 467.2 | 13.338 | 1.717 | 179.2 |
| model_execution | long_prefill | native_kv | 133.7 | 7.544 | 7.432 | 118.8 |
| replay_e2e | long_prefill | native_kv | 163.6 | 5.016 | 6.136 | 118.8 |
| model_execution | long_prefill | paged_eager | 135.8 | 12.219 | 7.336 | 175.9 |
| replay_e2e | long_prefill | paged_eager | 134.2 | 9.398 | 7.458 | 175.9 |
| model_execution | long_prefill | paged_graph | 925.2 | 12.812 | 0.945 | 184.0 |
| replay_e2e | long_prefill | paged_graph | 502.3 | 13.833 | 1.743 | 184.0 |
| model_execution | page_growth | native_kv | 1088.5 | 6.218 | 7.304 | 192.6 |
| replay_e2e | page_growth | native_kv | 1096.3 | 7.201 | 7.218 | 192.6 |
| model_execution | page_growth | paged_eager | 1175.9 | 7.702 | 6.558 | 198.6 |
| replay_e2e | page_growth | paged_eager | 1185.3 | 10.097 | 6.683 | 198.6 |
| model_execution | page_growth | paged_graph | 5833.4 | 10.611 | 1.089 | 206.8 |
| replay_e2e | page_growth | paged_graph | 3459.8 | 13.113 | 1.967 | 206.8 |
