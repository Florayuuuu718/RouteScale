"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from ddp_windows import GlobalWindowScheduler
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# reproducible single-GPU benchmark settings
benchmark = False
benchmark_warmup_steps = 20
benchmark_measure_steps = 100
benchmark_data_seed = 20260920
benchmark_results_dir = 'results/b1_single_gpu'
benchmark_run_id = 'run1'
# deterministic C-stage DDP benchmark settings
ddp_benchmark = False
ddp_scaling_mode = 'strong' # 'strong' or 'weak'
ddp_benchmark_warmup_steps = 20
ddp_benchmark_measure_steps = 100
ddp_data_seed = 20260920
ddp_tail_policy = 'drop' # 'drop' or 'pad'
ddp_results_dir = 'results/c_ddp'
ddp_run_id = 'run1'
# short PyTorch Profiler trace settings
profiler = False
profiler_startup_warmup_steps = 20
profiler_wait_steps = 2
profiler_warmup_steps = 2
profiler_active_steps = 10
profiler_data_seed = 20260920
profiler_trace_dir = 'logs/b2_profiler'
profiler_results_dir = 'results/b2_profiler'
profiler_run_id = 'baseline'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if sum((benchmark, ddp_benchmark, profiler)) > 1:
    raise ValueError('benchmark, ddp_benchmark, and profiler are mutually exclusive')
if ddp_benchmark and ddp_scaling_mode not in ('strong', 'weak'):
    raise ValueError("ddp_scaling_mode must be 'strong' or 'weak'")
if ddp:
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    if backend == 'nccl':
        init_process_group(backend=backend, device_id=torch.device(device))
    else:
        init_process_group(backend=backend)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    if not (ddp_benchmark and ddp_scaling_mode == 'weak'):
        # Standard training and strong scaling keep global work fixed, so divide
        # the configured accumulation count across ranks. Weak scaling keeps the
        # configured accumulation count on every rank.
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    ddp_local_rank = 0
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
data_generator = None
if benchmark or profiler:
    data_generator = torch.Generator(device='cpu')
    data_generator.manual_seed(
        benchmark_data_seed if benchmark else profiler_data_seed
    )

def record_region(name):
    if profiler:
        return torch.profiler.record_function(name)
    return nullcontext()

def get_batch(split, return_indices=False, indices=None):
    with record_region('get_batch'):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        if indices is None:
            ix = torch.randint(
                len(data) - block_size,
                (batch_size,),
                generator=data_generator,
            )
        else:
            ix = torch.as_tensor(indices, dtype=torch.int64, device='cpu')
            if ix.shape != (batch_size,):
                raise ValueError(
                    f'expected {batch_size} scheduled offsets, got shape {tuple(ix.shape)}'
                )
            if ix.min().item() < 0 or ix.max().item() + block_size >= len(data):
                raise ValueError('scheduled training window is outside the token file')
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
    if return_indices:
        return x, y, ix
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    state_dict = None
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
if device_type == 'cuda':
    torch.cuda.empty_cache()

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
raw_model = model.module if ddp else model # unwrap DDP container if needed

def run_training_step(X, Y, batch_indices=None, offsets_hasher=None):
    """Run one optimizer update and prefetch the next batch."""
    for micro_step in range(gradient_accumulation_steps):
        if offsets_hasher is not None:
            if batch_indices is None:
                raise RuntimeError('benchmark batch indices are missing')
            offsets = np.asarray(batch_indices.numpy(), dtype='<i8')
            offsets_hasher.update(offsets.tobytes())
        if ddp:
            # only synchronize gradients on the final accumulation micro-step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with record_region('forward'):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
        if offsets_hasher is None:
            X, Y = get_batch('train')
        else:
            X, Y, batch_indices = get_batch('train', return_indices=True)
        with record_region('backward'):
            scaler.scale(loss).backward()
    if grad_clip != 0.0:
        with record_region('gradient_clipping'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    with record_region('optimizer_step'):
        scaler.step(optimizer)
        scaler.update()
    with record_region('zero_grad'):
        optimizer.zero_grad(set_to_none=True)
    return X, Y, batch_indices, loss


def get_scheduled_batch(scheduler, update, micro_step):
    """Load one rank-local batch from the deterministic global window plan."""
    window_ids = scheduler.rank_window_ids(update)[micro_step]
    offsets = window_ids * block_size
    X, Y = get_batch('train', indices=offsets)
    return X, Y, window_ids


def run_scheduled_training_step(
    X, Y, window_ids, update, scheduler, window_hasher
):
    """Run an update while preserving B1's forward/next-batch/backward pipeline."""
    micro_losses = []
    for micro_step in range(gradient_accumulation_steps):
        expected_ids = scheduler.rank_window_ids(update)[micro_step]
        if not np.array_equal(window_ids, expected_ids):
            raise RuntimeError('scheduled batch cursor is out of sync')
        window_hasher.update(
            np.asarray(window_ids, dtype='<i8').tobytes(order='C')
        )
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, unscaled_loss = model(X, Y)
            loss = unscaled_loss / gradient_accumulation_steps
        micro_losses.append(unscaled_loss.detach())

        if micro_step + 1 < gradient_accumulation_steps:
            next_update = update
            next_micro_step = micro_step + 1
        else:
            next_update = update + 1
            next_micro_step = 0
        X, Y, window_ids = get_scheduled_batch(
            scheduler, next_update, next_micro_step
        )
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    mean_micro_loss = torch.stack(micro_losses).mean()
    return X, Y, window_ids, mean_micro_loss


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args):
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def run_b1_benchmark():
    if ddp:
        raise RuntimeError('B1 is a single-GPU benchmark; launch without torchrun')
    if device_type != 'cuda':
        raise RuntimeError('B1 requires a CUDA device for CUDA Event timing')
    if benchmark_warmup_steps < 1 or benchmark_measure_steps < 1:
        raise ValueError('benchmark warmup and measurement steps must be positive')

    print(
        f'B1 benchmark: {benchmark_warmup_steps} warmup updates, '
        f'{benchmark_measure_steps} measured updates, run {benchmark_run_id}'
    )
    X, Y, batch_indices = get_batch('train', return_indices=True)
    warmup_offsets = hashlib.sha256()
    for step in range(benchmark_warmup_steps):
        lr = get_lr(step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        X, Y, batch_indices, loss = run_training_step(
            X, Y, batch_indices, warmup_offsets
        )

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_offsets = hashlib.sha256()
    start_events = []
    end_events = []
    measured_losses = []
    for measured_step in range(benchmark_measure_steps):
        global_step = benchmark_warmup_steps + measured_step
        lr = get_lr(global_step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        X, Y, batch_indices, loss = run_training_step(
            X, Y, batch_indices, measured_offsets
        )
        end_event.record()
        start_events.append(start_event)
        end_events.append(end_event)
        measured_losses.append(loss.detach() * gradient_accumulation_steps)

    torch.cuda.synchronize(device)
    step_times_ms = [
        start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    step_losses = [loss_value.item() for loss_value in measured_losses]
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    median_step_ms = median(step_times_ms)
    total_measured_tokens = benchmark_measure_steps * tokens_per_iter
    manifest_path = Path(data_dir) / 'manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    tracked_files = ['train.py', 'model.py']
    config_path = Path('config/benchmark_tinystories.py')
    if config_path.exists():
        tracked_files.append(str(config_path))
    source_hashes = {
        path: sha256_file(path) for path in tracked_files if Path(path).exists()
    }
    git_diff = command_output(['git', 'diff', '--binary', 'HEAD']) or ''
    result = {
        'schema_version': 1,
        'benchmark': 'B1 single-GPU no-Profiler baseline',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': benchmark_run_id,
        'version': {
            'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
            'git_status': command_output(['git', 'status', '--short']),
            'tracked_diff_sha256': hashlib.sha256(git_diff.encode()).hexdigest(),
            'source_sha256': source_hashes,
        },
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'pytorch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(device),
            'driver': command_output([
                'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'
            ]),
        },
        'data': {
            'dataset': dataset,
            'manifest': manifest,
            'data_seed': benchmark_data_seed,
            'warmup_offsets_sha256': warmup_offsets.hexdigest(),
            'measurement_offsets_sha256': measured_offsets.hexdigest(),
            'warmup_window_count': (
                benchmark_warmup_steps * gradient_accumulation_steps * batch_size
            ),
            'measurement_window_count': (
                benchmark_measure_steps * gradient_accumulation_steps * batch_size
            ),
        },
        'configuration': {
            **config,
            'effective_gradient_accumulation_steps': gradient_accumulation_steps,
            'world_size': ddp_world_size,
            'tokens_per_update': tokens_per_iter,
            'model_parameter_count': sum(p.numel() for p in raw_model.parameters()),
            'non_position_embedding_parameter_count': raw_model.get_num_params(),
        },
        'measurement': {
            'timing': 'per-update CUDA Events; one synchronization after measurement',
            'profiler_enabled': False,
            'warmup_steps': benchmark_warmup_steps,
            'measured_steps': benchmark_measure_steps,
            'total_measured_tokens': total_measured_tokens,
            'step_times_ms': step_times_ms,
            'last_micro_batch_losses': step_losses,
            'all_losses_finite': all(math.isfinite(value) for value in step_losses),
            'median_step_ms': median_step_ms,
            'mean_step_ms': mean(step_times_ms),
            'min_step_ms': min(step_times_ms),
            'max_step_ms': max(step_times_ms),
            'tokens_per_second_from_median': tokens_per_iter / (median_step_ms / 1000),
            'aggregate_tokens_per_second': (
                total_measured_tokens / (sum(step_times_ms) / 1000)
            ),
            'first_loss': step_losses[0],
            'last_loss': step_losses[-1],
            'peak_allocated_mib': peak_allocated,
            'peak_reserved_mib': peak_reserved,
            'device_total_mib': total_memory,
        },
    }
    results_dir = Path(benchmark_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f'{benchmark_run_id}.json'
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(
        f'B1 result: median {median_step_ms:.3f} ms/update, '
        f'{result["measurement"]["tokens_per_second_from_median"]:,.0f} tokens/s, '
        f'peak {peak_allocated:.1f} MiB allocated, {peak_reserved:.1f} MiB reserved'
    )
    print(f'wrote {output_path}')


def run_c_ddp_benchmark():
    if device_type != 'cuda':
        raise RuntimeError('C DDP benchmark requires CUDA Event timing')
    if ddp_benchmark_warmup_steps < 1 or ddp_benchmark_measure_steps < 1:
        raise ValueError('C benchmark warmup and measurement steps must be positive')

    global_windows_per_update = (
        gradient_accumulation_steps * ddp_world_size * batch_size
    )
    train_path = Path(data_dir) / 'train.bin'
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    train_token_count = len(train_data)
    del train_data
    complete_window_count, trailing_token_count = divmod(
        train_token_count - 1, block_size
    )
    scheduler = GlobalWindowScheduler(
        window_count=complete_window_count,
        global_windows_per_update=global_windows_per_update,
        world_size=ddp_world_size,
        rank=ddp_rank,
        batch_size=batch_size,
        seed=ddp_data_seed,
        tail_policy=ddp_tail_policy,
    )
    if scheduler.local_micro_steps != gradient_accumulation_steps:
        raise RuntimeError('window plan does not match effective gradient accumulation')

    if master_process:
        print(
            f'C {ddp_scaling_mode} benchmark: world_size={ddp_world_size}, '
            f'{ddp_benchmark_warmup_steps} warmup updates, '
            f'{ddp_benchmark_measure_steps} measured updates, run {ddp_run_id}'
        )

    X, Y, window_ids = get_scheduled_batch(scheduler, 0, 0)
    warmup_window_hasher = hashlib.sha256()
    for update in range(ddp_benchmark_warmup_steps):
        lr = get_lr(update) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        X, Y, window_ids, loss = run_scheduled_training_step(
            X, Y, window_ids, update, scheduler, warmup_window_hasher
        )

    torch.cuda.synchronize(device)
    if ddp:
        dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    measured_window_hasher = hashlib.sha256()
    start_events = []
    end_events = []
    measured_losses = []
    measurement_start_update = ddp_benchmark_warmup_steps
    for measured_step in range(ddp_benchmark_measure_steps):
        update = measurement_start_update + measured_step
        lr = get_lr(update) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        X, Y, window_ids, loss = run_scheduled_training_step(
            X, Y, window_ids, update, scheduler, measured_window_hasher
        )
        end_event.record()
        start_events.append(start_event)
        end_events.append(end_event)
        measured_losses.append(loss)

    torch.cuda.synchronize(device)
    local_step_times_ms = [
        start.elapsed_time(end) for start, end in zip(start_events, end_events)
    ]
    local_losses = [value.item() for value in measured_losses]
    local_record = {
        'rank': ddp_rank,
        'local_rank': ddp_local_rank,
        'gpu': torch.cuda.get_device_name(device),
        'step_times_ms': local_step_times_ms,
        'mean_micro_batch_losses': local_losses,
        'all_losses_finite': all(math.isfinite(value) for value in local_losses),
        'warmup_window_ids_sha256': warmup_window_hasher.hexdigest(),
        'measurement_window_ids_sha256': measured_window_hasher.hexdigest(),
        'expected_warmup_window_ids_sha256': scheduler.hash_updates(
            0, ddp_benchmark_warmup_steps, ddp_rank
        ),
        'expected_measurement_window_ids_sha256': scheduler.hash_updates(
            measurement_start_update, ddp_benchmark_measure_steps, ddp_rank
        ),
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        'device_total_mib': (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        ),
    }
    if (
        local_record['warmup_window_ids_sha256']
        != local_record['expected_warmup_window_ids_sha256']
        or local_record['measurement_window_ids_sha256']
        != local_record['expected_measurement_window_ids_sha256']
    ):
        raise RuntimeError('runtime window sequence differs from the planned sequence')

    if ddp:
        gathered_records = [None] * ddp_world_size if master_process else None
        dist.gather_object(local_record, gathered_records, dst=0)
    else:
        gathered_records = [local_record]

    if master_process:
        rank_records = sorted(gathered_records, key=lambda record: record['rank'])
        slowest_rank_step_times_ms = [
            max(record['step_times_ms'][step] for record in rank_records)
            for step in range(ddp_benchmark_measure_steps)
        ]
        global_step_losses = [
            mean(record['mean_micro_batch_losses'][step] for record in rank_records)
            for step in range(ddp_benchmark_measure_steps)
        ]
        median_step_ms = median(slowest_rank_step_times_ms)
        global_tokens_per_update = global_windows_per_update * block_size
        total_measured_tokens = (
            ddp_benchmark_measure_steps * global_tokens_per_update
        )
        measured_global_ids = np.concatenate([
            scheduler.global_window_ids(update)
            for update in range(
                measurement_start_update,
                measurement_start_update + ddp_benchmark_measure_steps,
            )
        ])
        repeated_measurement_windows = (
            len(measured_global_ids) - len(np.unique(measured_global_ids))
        )
        manifest_path = Path(data_dir) / 'manifest.json'
        manifest = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        config_path = Path('config/benchmark_ddp_tinystories.py')
        tracked_files = ['train.py', 'model.py', 'ddp_windows.py']
        if config_path.exists():
            tracked_files.append(str(config_path))
        source_hashes = {
            path: sha256_file(path) for path in tracked_files if Path(path).exists()
        }
        git_diff = command_output(['git', 'diff', '--binary', 'HEAD']) or ''
        tokens_per_second = global_tokens_per_update / (median_step_ms / 1000)
        result = {
            'schema_version': 1,
            'benchmark': 'C deterministic DDP no-Profiler benchmark',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'run_id': ddp_run_id,
            'scaling_mode': ddp_scaling_mode,
            'version': {
                'git_commit': command_output(['git', 'rev-parse', 'HEAD']),
                'git_status': command_output(['git', 'status', '--short']),
                'tracked_diff_sha256': hashlib.sha256(git_diff.encode()).hexdigest(),
                'source_sha256': source_hashes,
            },
            'environment': {
                'platform': platform.platform(),
                'python': platform.python_version(),
                'pytorch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'nccl': torch.cuda.nccl.version(),
                'cudnn': torch.backends.cudnn.version(),
                'driver': command_output([
                    'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'
                ]),
                'nvidia_smi_list': command_output(['nvidia-smi', '-L']),
                'nvidia_smi_topology': command_output(['nvidia-smi', 'topo', '-m']),
            },
            'data': {
                'dataset': dataset,
                'manifest': manifest,
                'manifest_sha256': (
                    sha256_file(manifest_path) if manifest_path.exists() else None
                ),
                'data_seed': ddp_data_seed,
                'window_definition': 'non-overlapping block_size-token windows',
                'train_token_count': train_token_count,
                'complete_window_count': complete_window_count,
                'trailing_token_count': trailing_token_count,
                'tail_policy': ddp_tail_policy,
                'epoch_plan': scheduler.epoch_metadata(0),
                'warmup_global_window_ids_sha256': scheduler.hash_updates(
                    0, ddp_benchmark_warmup_steps
                ),
                'measurement_global_window_ids_sha256': scheduler.hash_updates(
                    measurement_start_update, ddp_benchmark_measure_steps
                ),
                'warmup_window_count': (
                    ddp_benchmark_warmup_steps * global_windows_per_update
                ),
                'measurement_window_count': len(measured_global_ids),
                'measurement_repeated_window_count': repeated_measurement_windows,
            },
            'configuration': {
                **config,
                'world_size': ddp_world_size,
                'effective_gradient_accumulation_steps_per_rank': (
                    gradient_accumulation_steps
                ),
                'local_windows_per_update': scheduler.local_windows_per_update,
                'global_windows_per_update': global_windows_per_update,
                'local_tokens_per_update': (
                    scheduler.local_windows_per_update * block_size
                ),
                'global_tokens_per_update': global_tokens_per_update,
                'model_parameter_count': sum(p.numel() for p in raw_model.parameters()),
                'non_position_embedding_parameter_count': raw_model.get_num_params(),
            },
            'measurement': {
                'timing': (
                    'per-rank CUDA Events; one synchronization after measurement; '
                    'global step uses the slowest rank'
                ),
                'profiler_enabled': False,
                'warmup_steps': ddp_benchmark_warmup_steps,
                'measured_steps': ddp_benchmark_measure_steps,
                'total_measured_tokens': total_measured_tokens,
                'slowest_rank_step_times_ms': slowest_rank_step_times_ms,
                'global_mean_losses': global_step_losses,
                'all_losses_finite': all(
                    record['all_losses_finite'] for record in rank_records
                ),
                'all_window_hashes_match_plan': all(
                    record['measurement_window_ids_sha256']
                    == record['expected_measurement_window_ids_sha256']
                    for record in rank_records
                ),
                'median_slowest_rank_step_ms': median_step_ms,
                'mean_slowest_rank_step_ms': mean(slowest_rank_step_times_ms),
                'min_slowest_rank_step_ms': min(slowest_rank_step_times_ms),
                'max_slowest_rank_step_ms': max(slowest_rank_step_times_ms),
                'global_tokens_per_second_from_median': tokens_per_second,
                'per_gpu_tokens_per_second_from_median': (
                    tokens_per_second / ddp_world_size
                ),
                'aggregate_global_tokens_per_second': (
                    total_measured_tokens
                    / (sum(slowest_rank_step_times_ms) / 1000)
                ),
                'first_loss': global_step_losses[0],
                'last_loss': global_step_losses[-1],
                'per_rank': rank_records,
            },
        }
        results_path = (
            Path(ddp_results_dir)
            / ddp_scaling_mode
            / f'{ddp_world_size}gpu'
            / f'{ddp_run_id}.json'
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(
            f'C result: median slowest-rank step {median_step_ms:.3f} ms, '
            f'{tokens_per_second:,.0f} global tokens/s'
        )
        print(f'wrote {results_path}')

    if ddp:
        dist.barrier()


def run_b2_profiler():
    if ddp:
        raise RuntimeError('B2 local trace is single-GPU; launch without torchrun')
    if device_type != 'cuda':
        raise RuntimeError('B2 requires a CUDA device')
    schedule_values = (
        profiler_wait_steps,
        profiler_warmup_steps,
        profiler_active_steps,
    )
    if min(schedule_values) < 1 or profiler_startup_warmup_steps < 1:
        raise ValueError('Profiler schedule and startup warmup steps must be positive')

    print(
        f'B2 profiler: {profiler_startup_warmup_steps} unprofiled warmup updates, '
        f'schedule wait={profiler_wait_steps}, warmup={profiler_warmup_steps}, '
        f'active={profiler_active_steps}, run {profiler_run_id}'
    )
    X, Y = get_batch('train')
    for step in range(profiler_startup_warmup_steps):
        lr = get_lr(step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        X, Y, _, loss = run_training_step(X, Y)
    torch.cuda.synchronize(device)

    trace_dir = Path(profiler_trace_dir)
    results_dir = Path(profiler_results_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f'{profiler_run_id}_trace.json'
    table_path = results_dir / f'{profiler_run_id}_key_averages.txt'
    events_path = results_dir / f'{profiler_run_id}_key_averages.json'

    def trace_handler(prof):
        prof.export_chrome_trace(str(trace_path))
        averages = list(prof.key_averages())
        table_path.write_text(
            prof.key_averages().table(
                sort_by='self_device_time_total',
                row_limit=80,
                max_name_column_width=80,
            ) + '\n',
            encoding='utf-8',
        )
        event_rows = [
            {
                'name': event.key,
                'count': event.count,
                'self_cpu_time_us': event.self_cpu_time_total,
                'cpu_time_total_us': event.cpu_time_total,
                'self_device_time_us': event.self_device_time_total,
                'device_time_total_us': event.device_time_total,
                'self_cpu_memory_bytes': event.self_cpu_memory_usage,
                'self_device_memory_bytes': event.self_device_memory_usage,
            }
            for event in averages
        ]
        event_rows.sort(
            key=lambda event: event['self_device_time_us'], reverse=True
        )
        events_path.write_text(
            json.dumps(event_rows, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    scheduled_steps = sum(schedule_values)
    observed_losses = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=profiler_wait_steps,
            warmup=profiler_warmup_steps,
            active=profiler_active_steps,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for profile_step in range(scheduled_steps):
            global_step = profiler_startup_warmup_steps + profile_step
            lr = get_lr(global_step) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            X, Y, _, loss = run_training_step(X, Y)
            observed_losses.append(loss.detach() * gradient_accumulation_steps)
            prof.step()

    torch.cuda.synchronize(device)
    losses = [value.item() for value in observed_losses]
    metadata = {
        'schema_version': 1,
        'profile': 'B2 single-GPU PyTorch Profiler trace',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_id': profiler_run_id,
        'schedule': {
            'startup_unprofiled_warmup_steps': profiler_startup_warmup_steps,
            'wait_steps': profiler_wait_steps,
            'profiler_warmup_steps': profiler_warmup_steps,
            'active_steps': profiler_active_steps,
        },
        'configuration': {
            **config,
            'effective_gradient_accumulation_steps': gradient_accumulation_steps,
            'tokens_per_update': tokens_per_iter,
        },
        'environment': {
            'pytorch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device),
        },
        'correctness': {
            'all_losses_finite': all(math.isfinite(value) for value in losses),
            'first_observed_loss': losses[0],
            'last_observed_loss': losses[-1],
        },
        'memory': {
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        },
        'artifacts': {
            'trace': str(trace_path),
            'key_averages_table': str(table_path),
            'key_averages_json': str(events_path),
        },
        'warning': 'Profiler timings include instrumentation overhead and are not formal throughput.',
    }
    metadata_path = results_dir / f'{profiler_run_id}_metadata.json'
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'wrote trace {trace_path}')
    print(f'wrote summaries {table_path}, {events_path}, {metadata_path}')


if benchmark:
    run_b1_benchmark()
    raise SystemExit(0)

if ddp_benchmark:
    run_c_ddp_benchmark()
    if ddp:
        destroy_process_group()
    raise SystemExit(0)

if profiler:
    run_b2_profiler()
    raise SystemExit(0)

if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        # exclude evaluation and checkpoint I/O from the following training iteration time
        t0 = time.time()
    if iter_num == 0 and eval_only:
        break
    # iter_num is the number of completed optimizer updates at this point
    if iter_num >= max_iters:
        break

    # forward, backward and optimizer update
    X, Y, _, loss = run_training_step(X, Y)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

if master_process and device_type == 'cuda':
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    print(
        f"peak CUDA memory: {peak_allocated:.1f} MiB allocated, "
        f"{peak_reserved:.1f} MiB reserved, {total_memory:.1f} MiB total"
    )

if ddp:
    destroy_process_group()
