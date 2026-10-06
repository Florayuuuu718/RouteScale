# C0 environment preflight

`preflight_single_gpu.json` is the local validation of the admission script.
All required checks passed for the one-GPU host. Its warning status records
two expected facts: no CUDA Toolkit `nvcc` and a dirty development worktree.
The report was produced through one-rank `torchrun`, so the NCCL collective
code path and value check also ran locally.

On the rented host, run the script through four-rank `torchrun` with
`--expected-gpus=4 --strict`. That execution must replace local assumptions
with the actual GPU inventory, topology, data hashes and NCCL AllReduce check.
