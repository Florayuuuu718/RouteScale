"""Complete GPT F qualifier / gated benchmark, launched by prepare_f_training.py."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist

from ep_training import clip_global_grad_norm, install, parameter_groups, scaled_loss, synchronize
from model import GPT, GPTConfig
from scripts.f_ep_worker import cpu, grads, optimizer_snapshot, sha, weights


def make_model(config, tiny, mode, device):
    torch.manual_seed(config["training"]["initialization_seed"])
    settings = (dict(n_layer=2, n_head=2, n_embd=16, vocab_size=64, block_size=8, bias=True, dropout=0)
                if tiny else dict(config["model"]))
    model = GPT(GPTConfig(**settings, moe_num_experts=4, moe_layer_index=1 if tiny else 4,
                          moe_capacity_factor=1.25, moe_drop_tokens=True, moe_balance_loss_weight=0.01)).to(device)
    install(model, mode)
    return model


def windows(config, tiny, updates, accumulation):
    generator = torch.Generator().manual_seed(config["training"]["data_seed"])
    batch, block = (4, 8) if tiny else (8, 512)
    if tiny:
        data = np.arange(10007, dtype=np.uint16) % 64
    else:
        path = ROOT / "data/tinystories_full/train.bin"
        if sha(path) != config["data"]["train_sha256"]:
            raise ValueError("TinyStories train hash mismatch")
        data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = torch.randint(len(data) - block, (updates, accumulation, batch), generator=generator).numpy()
    return data, starts, block, batch


def update(model, optimizer, mode, device, data, starts, block, batch, dtype, accumulation, clip, capture=True):
    rank, world = dist.get_rank(), dist.get_world_size()
    if batch % world:
        raise ValueError("global microbatch must divide world size")
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    sample = None
    for offsets in starts:
        selected = offsets if mode == "reference" else offsets[rank * (batch // world):(rank + 1) * (batch // world)]
        x = torch.from_numpy(np.stack([np.asarray(data[i:i + block], dtype=np.int64) for i in selected])).to(device)
        targets = torch.from_numpy(np.stack([np.asarray(data[i + 1:i + 1 + block], dtype=np.int64) for i in selected])).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext():
            logits, loss = model(x, targets)
            if mode != "reference":
                loss = scaled_loss(model, logits, targets, batch * block)
        (loss / accumulation).backward()
        total_loss += loss.detach() / accumulation
        if capture and sample is None:
            # Same actual token positions per rank; retain full vocab vectors.
            sample_logits = logits[rank * (batch // world):(rank + 1) * (batch // world)] if mode == "reference" else logits
            sample = cpu(sample_logits[:, :2])
    synchronize(model, mode)
    norm = clip_global_grad_norm(*parameter_groups(model, mode), clip)
    if mode != "reference":
        dist.all_reduce(total_loss)
    optimizer.step()
    if not capture:
        return {"loss": total_loss, "norm": norm}
    return {"loss": cpu(total_loss), "norm": cpu(norm), "sample_logits": sample}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--benchmark", choices=("reference", "replicated", "ep"))
    args = parser.parse_args()
    config = json.loads((ROOT / "config/f_ep_rental_requirements.json").read_text())
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if args.dtype == "bfloat16" and device.type != "cuda":
        raise ValueError("BF16 requires CUDA")
    if not args.tiny and args.dtype != "bfloat16":
        raise ValueError("full-shape scope is frozen to BF16")
    if device.type == "cuda":
        if torch.cuda.device_count() < world:
            raise ValueError("a distinct GPU per rank is required")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=60))
    training = config["training"]
    accumulation = 2 if args.tiny else training["accumulation_steps"]
    warmup = training["warmup_updates"] if args.benchmark else 0
    updates = warmup + training["measured_updates"] if args.benchmark else 1
    data, starts, block, batch = windows(config, args.tiny, updates, accumulation)
    entries = []
    try:
        for mode in ([args.benchmark] if args.benchmark else ["reference", "replicated", "ep"]):
            if args.benchmark == "reference" and world != 1:
                raise ValueError("formal reference timing must use one GPU")
            model = make_model(config, args.tiny, mode, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                         betas=tuple(training["betas"]), weight_decay=training["weight_decay"],
                                         foreach=False, fused=False)
            times = []
            for step in range(updates):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                result = update(model, optimizer, mode, device, data, starts[step], block, batch,
                                args.dtype, accumulation, 0.01 if args.tiny else training["global_grad_clip"], capture=not args.benchmark)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                duration = torch.tensor(time.perf_counter() - begin, device=device)
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                if step >= warmup:
                    times.append(float(duration))
            if args.benchmark:
                entry = {"mode": mode, "update_seconds": times, "warmup": warmup,
                         "last_loss": float(result["loss"]), "last_global_grad_norm": float(result["norm"])}
            else:
                result.update(gradients=grads(model), weights=weights(model), optimizer=optimizer_snapshot(model, optimizer))
                raw = args.output / f"{mode}_rank{rank}.pt"
                torch.save(result, raw)
                entry = {"mode": mode, "raw": str(raw.resolve()), "sha256": sha(raw)}
                del result
            entries.append(entry)
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report = {"rank": rank, "world": world, "dtype": args.dtype, "tiny": args.tiny,
                  "accumulation": accumulation, "global_tokens_per_update": batch * block * accumulation,
                  "window_sha256": hashlib.sha256(starts.tobytes()).hexdigest(), "entries": entries,
                  "torch": torch.__version__, "peak_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
