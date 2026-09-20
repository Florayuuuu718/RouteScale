# Local single-GPU candidate for the formal TinyStories experiments.
# Re-check and freeze the final batch size on the target four-GPU machine.

out_dir = "out-tinystories"
eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

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
max_iters = 300
lr_decay_iters = 300
min_lr = 3e-5
warmup_iters = 20

device = "cuda"
dtype = "bfloat16"
compile = False
