# Shared-backend correctness smoke

`trajectory_comparison.json` compares native PyTorch, DeepSpeed ZeRO-0 and
FSDP2 over four identical updates. Loss histories and global-window hashes are
identical. Native versus ZeRO-0 final parameters differ by at most
`7.451e-09`, within the declared `1e-7` tolerance.

The comparison uses a deliberately small float32 model to isolate framework
wiring from BF16 rounding and performance. Its timings are not benchmarks.
