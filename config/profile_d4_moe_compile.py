# D4 diagnostic trace for the selected torch.compile Top-1 MoE.

out_dir = "out-d4-moe-profile"
eval_interval = 1000000
eval_iters = 20
log_interval = 1000000
always_save_checkpoint = False
save_checkpoint = False
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

moe_num_experts = 4
moe_layer_index = 4
moe_capacity_factor = 1.25
moe_drop_tokens = True
moe_balance_loss_weight = 0.01

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
compile = True

benchmark = False
profiler = True
profiler_startup_warmup_steps = 20
profiler_wait_steps = 2
profiler_warmup_steps = 2
profiler_active_steps = 10
profiler_data_seed = 20260920
profiler_trace_dir = "logs/d4_moe_compile"
profiler_results_dir = "results/d4_moe_compile/profiler"
profiler_run_id = "moe_compiled"
