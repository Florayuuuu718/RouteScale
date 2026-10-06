# B2: short single-GPU PyTorch Profiler trace of the frozen B1 workload.

out_dir = "out-b2-profiler"
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

device = "cuda"
dtype = "bfloat16"
compile = False

benchmark = False
profiler = True
profiler_startup_warmup_steps = 20
profiler_wait_steps = 2
profiler_warmup_steps = 2
profiler_active_steps = 10
profiler_data_seed = 20260920
profiler_trace_dir = "logs/b2_profiler"
profiler_results_dir = "results/b2_profiler"
profiler_run_id = "baseline"
