# D3 deterministic 300-update Dense stability run.

out_dir = "out-d3-dense-stability"
eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = False
save_checkpoint = False

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
moe_metrics_path = "results/d3_moe/stability_dense.json"
moe_data_seed = 20260920
metrics_experiment_name = "D3 Dense 300-update stability"
metrics_config_path = "config/train_d3_dense_stability.py"

learning_rate = 3e-4
max_iters = 300
lr_decay_iters = 300
min_lr = 3e-5
warmup_iters = 20

device = "cuda"
dtype = "bfloat16"
compile = False
