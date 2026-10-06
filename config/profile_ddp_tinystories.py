# C4: short four-rank DDP communication trace.
#
# This trace explains collective/compute behavior. It is deliberately separate
# from the no-Profiler C2/C3 throughput benchmark.

out_dir = "out-c4-ddp-profiler"
eval_interval = 1000000
eval_iters = 20
log_interval = 1000000
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

wandb_log = False

dataset = "tinystories_full"
gradient_accumulation_steps = 4
batch_size = 8
block_size = 512

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

learning_rate = 3e-4
max_iters = 34
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 20
lr_decay_iters = 34
min_lr = 3e-5

backend = "nccl"
device = "cuda"
dtype = "bfloat16"
compile = False

benchmark = False
ddp_benchmark = False
profiler = True
profiler_distributed = True
profiler_startup_warmup_steps = 20
profiler_wait_steps = 2
profiler_warmup_steps = 2
profiler_active_steps = 10
profiler_data_seed = 20260920
profiler_tail_policy = "drop"
profiler_trace_dir = "logs/c_ddp_profiler"
profiler_results_dir = "results/c_ddp/profiler"
profiler_run_id = "4gpu"
