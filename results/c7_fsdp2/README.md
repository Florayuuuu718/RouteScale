# C7 FSDP2 local smoke

The model is wrapped from Transformer blocks up to the root with FSDP2
`fully_shard`. `single_gpu/result.json` records a two-update DCP save followed
by a new-process restore and continuation to update 4. Binary DCP shards are
ignored by Git.

The result validates model/optimizer state loading and deterministic window
progress. ReduceScatter, AllGather, sharded-memory reduction and performance
are explicitly pending a four-GPU run.

Formal server measurements use `train_fsdp2.py --benchmark`, sharing the same
no-validation/no-checkpoint timing driver and result schema as DeepSpeed. This
keeps the 20-warmup/100-measured protocol, slowest-rank timing, window hashes
and fused-optimizer choice directly auditable.
