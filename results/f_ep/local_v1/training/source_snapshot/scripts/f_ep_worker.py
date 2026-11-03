"""torchrun worker for the frozen F local numerical preparation scope."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from expert_parallel import ExpertParallelMoE, replicated_moe_control, sum_replicated_gradients
from moe import Top1MoE


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cpu(tensor):
    return tensor.detach().cpu().clone()


def optimizer_snapshot(model, optimizer):
    return {name: {k: cpu(v) if torch.is_tensor(v) else v for k, v in optimizer.state.get(p, {}).items()}
            for name, p in model.named_parameters()}


def grads(model):
    return {name: None if p.grad is None else cpu(p.grad) for name, p in model.named_parameters()}


def weights(model):
    return {name: cpu(p) for name, p in model.named_parameters()}


def fixture(case, step, total, width, world, seed, device, dtype):
    generator = torch.Generator().manual_seed(seed + step)
    x = torch.randn(total, width, generator=generator, dtype=dtype)
    routes = torch.arange(total) % 4
    if case in ("all_to_one", "capacity_observe", "capacity_drop"):
        routes[:] = 0
    if case == "changing_routes" and step > 0:
        routes[:] = step - 1
    x[:, :4] = -1
    x[torch.arange(total), routes] = 1
    cotangent = torch.randn(total, width, generator=generator, dtype=dtype)
    if case == "empty_origin" and world > 1:
        sizes = [0] + [total // (world - 1)] * (world - 1)
    elif case == "uneven" and world > 1:
        sizes = [1] * (world - 1) + [total - world + 1]
    else:
        sizes = [total // world] * world
    sizes[-1] += total - sum(sizes)
    return x.to(device), cotangent.to(device), sizes


def autocast(dtype, device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()


def run_case(args, config, case, rank, world, device):
    width = config["formal_shape"]["width"] if args.formal else config["width"]
    total = config["formal_shape"]["tokens"] if args.formal else 23
    updates = 1 if args.formal else config["updates"]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    factor = 1.25 if args.formal else (0.5 if case.startswith("capacity_") else 0.0)
    drop = args.formal or case == "capacity_drop"
    torch.manual_seed(config["seed"])
    source = Top1MoE(width, 4, bias=not args.formal, dropout=0,
                     capacity_factor=factor, drop_tokens=drop).to(device=device, dtype=dtype)
    with torch.no_grad():
        source.router.proj.weight.zero_()
        source.router.proj.weight[:, :4].copy_(5 * torch.eye(4, device=device, dtype=dtype))
        if source.router.proj.bias is not None:
            source.router.proj.bias.zero_()
    reference = copy.deepcopy(source)
    replicated = copy.deepcopy(source)
    ep = ExpertParallelMoE(source)
    opt_config = dict(config["optimizer"], foreach=False, fused=False)
    opt_config["betas"] = tuple(opt_config["betas"])
    optimizers = [torch.optim.AdamW(m.parameters(), **opt_config) for m in (reference, replicated, ep)]
    records = []
    for step in range(updates):
        x, cotangent, sizes = fixture(case, step, total, width, world, config["seed"], device, dtype)
        start, stop = sum(sizes[:rank]), sum(sizes[:rank + 1])
        ref_x = x.detach().clone().requires_grad_()
        ep_x = x[start:stop].detach().clone().requires_grad_()
        rep_x = x[start:stop].detach().clone().requires_grad_()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with autocast(args.dtype, device):
            ref_y, ref_routing = reference(ref_x, return_routing=True)
            ref_loss = (ref_y * cotangent).sum() / total + config["balance_weight"] * ref_routing.balance_loss
            ep_y = ep(ep_x)
            routing = ep.last_routing
            ep_loss = (ep_y * cotangent[start:stop]).sum() / total + config["balance_weight"] * routing.balance_loss
            rep_y, rep_balance = replicated_moe_control(
                replicated, rep_x, ref_routing.expert_counts, ~ref_routing.dropped_mask[start:stop]
            )
            rep_loss = (rep_y * cotangent[start:stop]).sum() / total + config["balance_weight"] * rep_balance
        ref_loss.backward()
        ep_loss.backward()
        ep.synchronize_router_gradients()
        rep_loss.backward()
        sum_replicated_gradients(replicated.parameters())
        record = {
            "step": step, "sizes": sizes, "total": total, "width": width,
            "rank": rank, "capacity": routing.capacity,
            "send": routing.send_counts, "receive": routing.receive_counts,
            "actual": {"output": cpu(ep_y), "input_grad": cpu(ep_x.grad),
                       "gradients": grads(ep), "counts": cpu(routing.global_counts),
                       "processed": cpu(routing.global_processed_counts),
                       "indices": cpu(routing.expert_indices), "drops": cpu(routing.dropped_mask)},
            "replicated": {"output": cpu(rep_y), "input_grad": cpu(rep_x.grad), "gradients": grads(replicated)},
            "reference": {"output": cpu(ref_y[start:stop]), "input_grad": cpu(ref_x.grad[start:stop]),
                          "gradients": grads(reference), "counts": cpu(ref_routing.expert_counts),
                          "processed": cpu(ref_routing.processed_counts),
                          "indices": cpu(ref_routing.expert_indices[start:stop]),
                          "drops": cpu(ref_routing.dropped_mask[start:stop])},
        }
        for model, optimizer, label in zip((reference, replicated, ep), optimizers, ("reference", "replicated", "actual")):
            optimizer.step()
            record[label]["weights"] = weights(model)
            record[label]["optimizer"] = optimizer_snapshot(model, optimizer)
        records.append(record)
    path = args.output / f"{case}_rank{rank}.pt"
    torch.save(records, path)
    return {"case": case, "rank": rank, "steps": updates, "raw": str(path.resolve()), "sha256": sha(path),
            "owned_experts": list(ep.experts), "owned_parameters": sum(p.numel() for p in ep.parameters()),
            "replicated_parameters": sum(p.numel() for p in reference.parameters())}


def gpt_smoke(args, device):
    """Small full-model integration on a fixed global batch, including shared gradients."""
    from model import GPT, GPTConfig
    world, rank = dist.get_world_size(), dist.get_rank()
    if 4 % world:
        raise ValueError("GPT smoke global batch4 must divide world size")
    torch.manual_seed(20261103)
    ref = GPT(GPTConfig(block_size=8, vocab_size=64, n_layer=2, n_head=2,
                       n_embd=16, bias=True, dropout=0, moe_num_experts=4,
                       moe_layer_index=1, moe_capacity_factor=1.25,
                       moe_drop_tokens=True, moe_balance_loss_weight=0.01)).to(device)
    actual = copy.deepcopy(ref)
    actual.transformer.h[1].mlp = ExpertParallelMoE(actual.transformer.h[1].mlp)
    ids = torch.arange(32, device=device).reshape(4, 8)
    target = (ids + 1) % 64
    records = []
    start, stop = rank * (4 // world), (rank + 1) * (4 // world)
    for model in (ref, actual):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.1, foreach=False, fused=False)
        with autocast(args.dtype, device):
            if model is ref:
                logits, loss = model(ids, target)
            else:
                logits, _ = model(ids[start:stop], target[start:stop])
                loss = (torch.nn.functional.cross_entropy(logits.reshape(-1, 64), target[start:stop].reshape(-1), reduction="sum")
                        / ids.numel() + 0.01 * model.transformer.h[1].mlp.last_routing.balance_loss)
        loss.backward()
        if model is actual:
            sum_replicated_gradients(p for name, p in model.named_parameters()
                                     if not name.startswith("transformer.h.1.mlp.experts."))
            total_loss = loss.detach().clone()
            dist.all_reduce(total_loss)
        else:
            total_loss = loss.detach()
            logits = logits[start:stop]
        record = {"logits": cpu(logits), "loss": cpu(total_loss), "gradients": grads(model)}
        optimizer.step()
        record["weights"] = weights(model)
        record["optimizer"] = optimizer_snapshot(model, optimizer)
        records.append(record)
    path = args.output / f"gpt_rank{rank}.pt"
    torch.save({"reference": records[0], "actual": records[1]}, path)
    return {"case": "gpt", "rank": rank, "steps": 1, "raw": str(path.resolve()), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float64", "float32", "bfloat16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--gpt", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/f_ep_matrix.json").read_text())
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if args.dtype == "bfloat16" and device.type != "cuda":
        raise ValueError("BF16 scope requires CUDA")
    if device.type == "cuda":
        if torch.cuda.device_count() < world:
            raise RuntimeError("one distinct GPU per CUDA rank is required")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend, timeout=timedelta(seconds=config["group_timeout_s"]))
    try:
        cases = ["balanced"] if args.formal or args.trace else config["cases"]
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities) if args.trace else nullcontext()
        with profiler:
            entries = ([gpt_smoke(args, device)] if args.gpt else
                       [run_case(args, config, case, rank, world, device) for case in cases])
        if args.trace:
            trace = args.output / f"trace_rank{rank}.json"
            profiler.export_chrome_trace(str(trace))
            events = [{"key": event.key, "count": event.count, "cpu_time_total_us": event.cpu_time_total,
                       "self_cpu_time_total_us": event.self_cpu_time_total}
                      for event in profiler.key_averages()
                      if "alltoall" in event.key.lower() or event.key.startswith("ep/")]
        else:
            trace, events = None, []
        report = {"rank": rank, "world": world, "backend": backend, "device": str(device),
                  "dtype": args.dtype, "torch": torch.__version__, "python": platform.python_version(),
                  "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version() if device.type == "cuda" else None,
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
                  "peak_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "entries": entries, "trace": str(trace) if trace else None, "events": events}
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
