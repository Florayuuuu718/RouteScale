# D2 mechanism-isolation experiment on TinyStories debug data.
# This configuration records routing evidence; it is not a throughput benchmark.

out_dir = "out-d2-moe-debug"
eval_interval = 1000
eval_iters = 2
log_interval = 1
always_save_checkpoint = False

wandb_log = False

dataset = "tinystories_debug"
gradient_accumulation_steps = 1
batch_size = 4
block_size = 256

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

moe_num_experts = 4
moe_layer_index = 2
moe_capacity_factor = 0.0
moe_drop_tokens = False
moe_balance_loss_weight = 0.0
moe_metrics_path = "results/d2_moe/unbounded.json"
moe_data_seed = 20260920

learning_rate = 6e-4
max_iters = 20
lr_decay_iters = 20
min_lr = 6e-5
warmup_iters = 2

device = "cuda"
dtype = "bfloat16"
compile = False
