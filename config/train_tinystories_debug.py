# Small local configuration for validating the TinyStories data and training path.
# It is not a formal performance benchmark configuration.

out_dir = "out-tinystories-debug"
eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

wandb_log = False

dataset = "tinystories_debug"
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 100
lr_decay_iters = 100
min_lr = 6e-5
warmup_iters = 10

device = "cuda"
dtype = "bfloat16"
compile = False
