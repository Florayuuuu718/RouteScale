"""PyTorch FSDP2 ``fully_shard`` entry point with DCP checkpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from ddp_windows import GlobalWindowScheduler
from distributed_common import (
    MemmapTokenDataset,
    atomic_json_dump,
    atomic_torch_save,
    autocast_context,
    barrier,
    build_adamw,
    build_model,
    capture_rng_state,
    destroy_distributed,
    environment_metadata,
    evaluate_loss,
    git_metadata,
    gather_rank_objects,
    initialize_distributed,
    restore_rng_state,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/tinystories_debug")
    parser.add_argument("--out-dir", default="results/c7_fsdp2/single_gpu")
    parser.add_argument("--max-updates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", default=None,
                        help="DCP checkpoint directory, e.g. .../update-00000002")
    parser.add_argument("--eval-windows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--global-windows-per-update", type=int, default=32)
    parser.add_argument("--tail-policy", choices=("drop", "pad"), default="pad")
    parser.add_argument("--data-seed", type=int, default=20260920)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--validation-seed", type=int, default=20260921)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--reshard-after-forward", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="nccl")
    return parser.parse_args()


def signature(args, dataset, scheduler, world_size: int) -> dict:
    return {
        "backend": "fsdp2",
        "data_manifest_sha256": dataset.manifest_sha256,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "global_windows_per_update": args.global_windows_per_update,
        "world_size": world_size,
        "tail_policy": args.tail_policy,
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "bias": args.bias,
        "dtype": args.dtype,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "reshard_after_forward": args.reshard_after_forward,
        "updates_per_epoch": scheduler.updates_per_epoch,
    }


def save_dcp_checkpoint(
    *, model, optimizer, checkpoint_dir: Path, metadata: dict, rank: int, master: bool
) -> None:
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        storage_writer=FileSystemWriter(checkpoint_dir, overwrite=True),
    )
    barrier()
    atomic_torch_save(capture_rng_state(), checkpoint_dir / f"rng_rank{rank}.pt")
    barrier()
    if master:
        atomic_json_dump(metadata, checkpoint_dir / "route_scale_metadata.json")
    barrier()


def load_dcp_checkpoint(*, model, optimizer, checkpoint_dir: Path, rank: int) -> dict:
    metadata_path = checkpoint_dir / "route_scale_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=checkpoint_dir)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=StateDictOptions(full_state_dict=False, cpu_offload=False),
    )
    rng_path = checkpoint_dir / f"rng_rank{rank}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(rng_path)
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return metadata


def main() -> None:
    args = parse_args()
    if args.max_updates <= 0:
        raise ValueError("max-updates must be positive")
    env = initialize_distributed(device=args.device, backend=args.backend)
    try:
        if not env.distributed or env.device_type != "cuda":
            raise RuntimeError(
                "FSDP2 requires a CUDA torchrun launch, including for one GPU: "
                "uv run torchrun --standalone --nproc_per_node=1 train_fsdp2.py"
            )
        seed_everything(args.model_seed)
        dataset = MemmapTokenDataset(args.data_dir, args.block_size)
        scheduler = GlobalWindowScheduler(
            window_count=dataset.train_window_count,
            global_windows_per_update=args.global_windows_per_update,
            world_size=env.world_size,
            rank=env.rank,
            batch_size=args.batch_size,
            seed=args.data_seed,
            tail_policy=args.tail_policy,
        )
        run_signature = signature(args, dataset, scheduler, env.world_size)
        model = build_model(
            block_size=args.block_size,
            vocab_size=dataset.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=args.bias,
        )
        mesh = init_device_mesh(
            "cuda", (env.world_size,), mesh_dim_names=("data_parallel",)
        )
        param_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
        mixed_precision = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32 if param_dtype is not None else None,
            output_dtype=param_dtype,
        )
        # Shard transformer blocks first, then the root. Embedding/lm_head tied
        # weights remain owned by the same root FSDP group.
        for block in model.transformer.h:
            fully_shard(
                block,
                mesh=mesh,
                reshard_after_forward=args.reshard_after_forward,
                mp_policy=mixed_precision,
            )
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=mixed_precision,
        )
        optimizer = build_adamw(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            fused=False,
        )

        next_update = 0
        loss_history: list[float] = []
        initial_validation_loss = None
        if args.resume_from:
            metadata = load_dcp_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=Path(args.resume_from),
                rank=env.rank,
            )
            if metadata["signature"] != run_signature:
                raise ValueError("FSDP2 checkpoint signature mismatch")
            next_update = int(metadata["next_update"])
            loss_history = list(metadata["loss_history"])
            initial_validation_loss = float(metadata["initial_validation_loss"])
            if metadata["global_window_prefix_sha256"] != scheduler.hash_updates(0, next_update):
                raise ValueError("FSDP2 checkpoint data-window progress mismatch")

        validation_ids = dataset.fixed_validation_ids(
            args.eval_windows, args.validation_seed
        )
        if initial_validation_loss is None:
            initial_validation_loss = evaluate_loss(
                model,
                dataset,
                validation_ids,
                batch_size=args.batch_size,
                device=env.device,
                device_type=env.device_type,
                dtype=args.dtype,
            )
        if args.max_updates < next_update:
            raise ValueError("max-updates is behind the resumed checkpoint")
        if args.max_updates > scheduler.updates_per_epoch:
            raise ValueError("this entry point currently accepts at most one epoch")

        out_dir = Path(args.out_dir)
        invocation_start = next_update
        torch.cuda.reset_peak_memory_stats(env.device)
        torch.cuda.synchronize(env.device)
        started = time.perf_counter()
        last_checkpoint_update = None
        for update in range(next_update, args.max_updates):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro_step, window_ids in enumerate(scheduler.rank_window_ids(update)):
                is_last = micro_step + 1 == scheduler.local_micro_steps
                model.set_requires_gradient_sync(is_last)
                model.set_reshard_after_backward(is_last)
                x, y = dataset.batch("train", window_ids, env.device)
                with autocast_context(env.device_type, args.dtype):
                    _, loss = model(x, y)
                    scaled_loss = loss / scheduler.local_micro_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                scaled_loss.backward()
                micro_losses.append(loss.detach())
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            mean_loss = torch.stack(micro_losses).mean()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= env.world_size
            loss_history.append(float(mean_loss))
            next_update = update + 1
            if env.master:
                print(
                    f"FSDP2 update {next_update}/{args.max_updates} "
                    f"loss {loss_history[-1]:.6f}",
                    flush=True,
                )
            if args.checkpoint_every and next_update % args.checkpoint_every == 0:
                checkpoint_dir = out_dir / "checkpoints" / f"update-{next_update:08d}"
                save_dcp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    checkpoint_dir=checkpoint_dir,
                    metadata={
                        "format_version": 1,
                        "signature": run_signature,
                        "next_update": next_update,
                        "loss_history": loss_history,
                        "initial_validation_loss": initial_validation_loss,
                        "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                    },
                    rank=env.rank,
                    master=env.master,
                )
                last_checkpoint_update = next_update

        torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - started
        final_validation_loss = evaluate_loss(
            model,
            dataset,
            validation_ids,
            batch_size=args.batch_size,
            device=env.device,
            device_type=env.device_type,
            dtype=args.dtype,
        )
        checkpoint_dir = out_dir / "checkpoints" / f"update-{next_update:08d}"
        if last_checkpoint_update != next_update:
            save_dcp_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
                metadata={
                    "format_version": 1,
                    "signature": run_signature,
                    "next_update": next_update,
                    "loss_history": loss_history,
                    "initial_validation_loss": initial_validation_loss,
                    "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                },
                rank=env.rank,
                master=env.master,
            )

        rank_metrics = gather_rank_objects(
            {
                "rank": env.rank,
                "elapsed_seconds_this_invocation": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
            },
            env,
        )

        if env.master:
            invocation_updates = next_update - invocation_start
            slowest_elapsed = max(
                item["elapsed_seconds_this_invocation"] for item in rank_metrics
            )
            result = {
                "schema_version": 1,
                "stage": "C7",
                "backend": "fsdp2",
                "status": "smoke_complete",
                "configuration": vars(args),
                "checkpoint_signature": run_signature,
                "environment": environment_metadata(env),
                "git": git_metadata(),
                "dataset": {
                    "manifest_sha256": dataset.manifest_sha256,
                    "train_window_count": dataset.train_window_count,
                },
                "progress": {
                    "next_update": next_update,
                    "updates_per_epoch": scheduler.updates_per_epoch,
                    "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                },
                "metrics": {
                    "initial_validation_loss": initial_validation_loss,
                    "final_validation_loss": final_validation_loss,
                    "last_train_loss": loss_history[-1] if loss_history else None,
                    "updates_this_invocation": invocation_updates,
                    "elapsed_seconds_this_invocation": slowest_elapsed,
                    "per_rank": rank_metrics,
                    "tokens_per_second_this_invocation": (
                        invocation_updates
                        * args.global_windows_per_update
                        * args.block_size
                        / slowest_elapsed
                    ),
                    "peak_allocated_bytes": max(
                        item["peak_allocated_bytes"] for item in rank_metrics
                    ),
                    "peak_reserved_bytes": max(
                        item["peak_reserved_bytes"] for item in rank_metrics
                    ),
                },
                "artifacts": {"checkpoint_dir": str(checkpoint_dir)},
            }
            atomic_json_dump(result, out_dir / "result.json")
            print(f"saved {out_dir / 'result.json'}", flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
