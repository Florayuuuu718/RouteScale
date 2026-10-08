# D2 Top-1 MoE routing-mechanism results

Date: 2026-10-07. These experiments validate capacity accounting, deterministic
token drop, balance-loss wiring, and routing telemetry. They are short
TinyStories debug runs, not formal throughput or convergence results.

## Protocol

All four runs used one BF16 RTX 5060 Laptop GPU, one MoE block with four
experts, batch size 4, sequence length 256, 20 optimizer updates, seed
`20260920`, and `compile=False`. Every run consumed the same 80 training
windows with offsets SHA256
`f9cba2a564a65514bd1d19eb0d38a660231a8987c7d06224d64974a63063121b`.
Source hashes, model parameter count (17,653,248), optimizer, initialization,
and all non-mechanism configuration values matched.

The sequence changed one mechanism at a time:

```bash
uv run python train.py config/train_moe_d2_debug.py

uv run python train.py config/train_moe_d2_debug.py \
  --moe_capacity_factor=1.25 \
  --moe_metrics_path='results/d2_moe/capacity_observe.json'

uv run python train.py config/train_moe_d2_debug.py \
  --moe_capacity_factor=1.25 --moe_drop_tokens=True \
  --moe_metrics_path='results/d2_moe/capacity_drop.json'

uv run python train.py config/train_moe_d2_debug.py \
  --moe_capacity_factor=1.25 --moe_drop_tokens=True \
  --moe_balance_loss_weight=0.01 \
  --moe_metrics_path='results/d2_moe/balance_loss.json'

uv run python scripts/summarize_d2_moe.py
```

## Results

| Variant | Aggregate counts | Aggregate max/mean | Aggregate CV | Overflow | Dropped | Final max/mean | Final CV | Final drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unbounded | 8,326 / 3,669 / 2,586 / 5,899 | 1.626 | 0.430 | 0% | 0% | 2.797 | 1.066 | 0% |
| Capacity observed | 8,326 / 3,669 / 2,586 / 5,899 | 1.626 | 0.430 | 30.75% | 0% | 2.797 | 1.066 | 0% |
| Capacity + drop | 7,436 / 5,354 / 2,043 / 5,647 | 1.452 | 0.380 | 31.81% | 31.81% | 2.938 | 1.133 | 42.19% |
| Capacity + drop + balance | 5,589 / 4,819 / 5,628 / 4,444 | 1.099 | 0.099 | 19.47% | 19.47% | 1.316 | 0.289 | 1.66% |

Capacity observation was non-mutating: its expert counts, LM loss and total
loss matched the unbounded run update by update. This separates measurement
from enforcement.

Without balance loss, enforcing capacity changed the subsequent route because
dropped tokens no longer supplied task gradients through their expert branch.
Across the short run it dropped 31.81% of tokens, and the final update routed no
more evenly than the unbounded case. With balance-loss weight 0.01, aggregate
load CV fell to 0.099 and the final expert counts were 305 / 238 / 337 / 144.
Final drop rate fell from 42.19% to 1.66%, while final LM loss was 8.2632 versus
8.2658 without balance loss.

The balance run was not uniformly stable from its first update: its drop rate
temporarily reached 68.55% at update 4 before recovering. Twenty updates and a
two-batch initial validation estimate cannot establish better language-model
quality or long-run routing stability. D3 must use TinyStories full, longer
training, and the frozen three-run no-Profiler protocol.

Raw artifacts are `unbounded.json`, `capacity_observe.json`,
`capacity_drop.json`, and `balance_loss.json`; machine-validated aggregates are
in `summary.json`. All 34 repository tests passed. A separate checkpoint smoke
saved `capacity_factor=1.25`, token drop, and balance-loss weight 0.01, then a
new process restored the model, optimizer, and all three mechanism settings and
completed the next update with finite loss. The temporary checkpoint was
removed after verification.
