# D3 formal Dense baseline: single GPU, no Profiler, eager mode.

out_dir = "out-d3-dense-benchmark"
eval_interval = 1000000
eval_iters = 20
log_interval = 1000000
eval_only = False
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

moe_num_experts = 0
moe_layer_index = 4
moe_capacity_factor = 0.0
moe_drop_tokens = False
moe_balance_loss_weight = 0.0

learning_rate = 3e-4
max_iters = 120
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 20
lr_decay_iters = 120
min_lr = 3e-5

device = "cuda"
dtype = "bfloat16"
compile = False

benchmark = True
benchmark_warmup_steps = 20
benchmark_measure_steps = 100
benchmark_data_seed = 20260920
benchmark_results_dir = "results/d3_moe/benchmark/dense"
benchmark_run_id = "run1"
benchmark_name = "D3 Dense single-GPU no-Profiler benchmark"
benchmark_config_path = "config/benchmark_d3_dense.py"
