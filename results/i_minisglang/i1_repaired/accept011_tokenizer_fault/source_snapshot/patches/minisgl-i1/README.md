# Mini-SGLang I1 patch series

Apply the numbered patches in order to MIT-licensed upstream commit
`9a91cfafe754aa85daee49998176275667eb58f2`.
`series.json` records SHA256 hashes and the expected resulting Git tree.
The original upstream checkout stays clean; the preparation script creates an
independent worktree and refuses to reset or overwrite a dirty checkout.

```bash
runs/envs/mini-sglang/bin/python scripts/prepare_i1_lifecycle.py
```

Patches 0001–0006 repair frontend ownership, ordered cancellation cleanup,
finish reasons, context rejection, worker health and process shutdown.
Patch 0007 bounds admission through both backend completion and frontend close.
The repair supports the tested shared tokenizer and non-overlap execution path;
startup rejects separate tokenizers or overlap rather than silently using an
unaccepted cancellation path. TP2/4 remains pending.

See [the implementation and runbook](../../docs/i1_lifecycle_repairs.md) and
[real-GPU evidence](../../results/i_minisglang/i1_repaired/README.md).
No runtime repairs are installed by the observation hooks.
