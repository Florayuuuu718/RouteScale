# C stage: deterministic 1/2/4-GPU DDP benchmark.
#
# Strong scaling keeps this configured global accumulation count fixed; train.py
# divides it across ranks. Weak scaling keeps all four micro-steps on every rank.

out_dir = "out-c-ddp-tinystories"
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
max_iters = 120
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 20
lr_decay_iters = 120
min_lr = 3e-5

backend = "nccl"
device = "cuda"
dtype = "bfloat16"
compile = False

benchmark = False
profiler = False
ddp_benchmark = True
ddp_scaling_mode = "strong"
ddp_benchmark_warmup_steps = 20
ddp_benchmark_measure_steps = 100
ddp_data_seed = 20260920
ddp_tail_policy = "drop"
ddp_results_dir = "results/c_ddp"
ddp_run_id = "run1"
