# D1 local correctness smoke test. This is not a formal performance result.

out_dir = "out-d1-moe-debug"
eval_interval = 5
eval_iters = 2
log_interval = 1
always_save_checkpoint = True

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

learning_rate = 6e-4
max_iters = 10
lr_decay_iters = 10
min_lr = 6e-5
warmup_iters = 2

device = "cuda"
dtype = "bfloat16"
compile = False
