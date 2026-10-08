# D1 single-GPU Top-1 MoE correctness evidence

Date: 2026-10-07. D1 establishes the minimal local Top-1 routing path. It is
not a formal Dense/MoE performance result.

## Automated correctness checks

The D1 test module covers:

- one route per token, expert counts, and restoration of token order;
- `num_experts=1` output, input-gradient, and parameter-gradient equivalence
  with the Dense MLP;
- finite, non-zero gradients for the router and selected experts, with no
  gradient for an unselected expert;
- BF16 autocast dispatch and output dtype;
- GPT `(logits, loss)` compatibility and finite loss;
- identical shared Dense/MoE initialization and strict MoE state-dict reload.

Run the D1 checks with:

```bash
uv run python -m unittest discover -s tests -p 'test_moe.py' -v
```

Run the complete repository regression suite with:

```bash
uv run python -m unittest discover -s tests -v
```

On 2026-10-07, all 7 D1 tests and all 26 repository tests passed.

## CUDA training smoke

The checked-in smoke configuration is
`config/train_moe_tinystories_debug.py`: one MoE block, four experts,
TinyStories debug data, BF16, and `compile=False`.

A ten-update run of the checked-in configuration on the NVIDIA GeForce RTX
5060 Laptop GPU completed without NaN or OOM. Observed training loss changed
from 10.8145 to 9.4836, while the two-batch validation estimate changed from
10.8219 to 9.3650. Peak CUDA memory was 929.8 MiB allocated and 1,210.0 MiB
reserved. These short-run loss values establish smoke stability only; they are
not a quality or performance comparison.

The run saved a 4-expert checkpoint, and a new process strictly loaded its
model and optimizer state and completed update 11 with finite loss.
The pre-D-stage Dense checkpoint in `out-tinystories` also loaded with the new
code and completed validation, confirming compatibility with checkpoints that
do not contain MoE configuration fields.

Capacity, token dropping, load-balancing loss, and routing telemetry were added
and isolated in D2. Formal three-run performance comparisons remain D3 work.
