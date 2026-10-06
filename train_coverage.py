"""C5 deterministic full-window training with native PyTorch DDP checkpointing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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
    evaluate_full_validation,
    evaluate_loss,
    gather_rank_objects,
    git_metadata,
    initialize_distributed,
    restore_rng_state,
    seed_everything,
    sha256_file,
    state_dict_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/tinystories_debug")
    parser.add_argument("--out-dir", default="results/c5_coverage/single_gpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-updates", type=int, default=0,
                        help="absolute update target; 0 means one complete epoch")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-windows", type=int, default=32)
    parser.add_argument(
        "--full-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate every validation target token before and after training",
    )
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
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto")
    return parser.parse_args()


def checkpoint_signature(args, dataset, scheduler, world_size: int) -> dict:
    return {
        "data_manifest_sha256": dataset.manifest_sha256,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "global_windows_per_update": args.global_windows_per_update,
        "world_size": world_size,
        "tail_policy": args.tail_policy,
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "validation_seed": args.validation_seed,
        "eval_windows": args.eval_windows,
        "full_validation": args.full_validation,
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
        "updates_per_epoch": scheduler.updates_per_epoch,
    }


def save_checkpoint(
    *, path: Path, raw_model, optimizer, scaler, next_update: int,
    loss_history: list[float], initial_validation_loss: float,
    signature: dict, scheduler: GlobalWindowScheduler, env,
) -> str | None:
    rng_by_rank = gather_rank_objects(capture_rng_state(), env)
    model_hash = None
    if env.master:
        model_state = raw_model.state_dict()
        model_hash = state_dict_sha256(model_state)
        atomic_torch_save(
            {
                "format_version": 1,
                "backend": "native_ddp",
                "signature": signature,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "next_update": next_update,
                "loss_history": loss_history,
                "initial_validation_loss": initial_validation_loss,
                "rng_by_rank": rng_by_rank,
                "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                "model_state_sha256": model_hash,
            },
            path,
        )
    barrier()
    return model_hash


def main() -> None:
    args = parse_args()
    if args.max_updates < 0 or args.checkpoint_every < 0:
        raise ValueError("update counts must be non-negative")
    env = initialize_distributed(device=args.device, backend=args.backend)
    try:
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
        signature = checkpoint_signature(args, dataset, scheduler, env.world_size)
        model = build_model(
            block_size=args.block_size,
            vocab_size=dataset.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            bias=args.bias,
        ).to(env.device)
        optimizer = build_adamw(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            fused=False,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=env.device_type == "cuda" and args.dtype == "float16"
        )

        next_update = 0
        loss_history: list[float] = []
        initial_validation_loss: float | None = None
        restored_rng = None
        resume_path = Path(args.resume_from) if args.resume_from else None
        if resume_path is not None:
            # RNG snapshots must remain CPU ByteTensors. Optimizer.load_state_dict
            # moves its tensors to the parameter device after this CPU load.
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            if checkpoint.get("format_version") != 1:
                raise ValueError("unsupported checkpoint format")
            if checkpoint["signature"] != signature:
                raise ValueError(
                    "checkpoint signature does not match this run:\n"
                    f"checkpoint={checkpoint['signature']}\ncurrent={signature}"
                )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            next_update = int(checkpoint["next_update"])
            loss_history = list(checkpoint["loss_history"])
            initial_validation_loss = float(checkpoint["initial_validation_loss"])
            expected_prefix = scheduler.hash_updates(0, next_update)
            if checkpoint["global_window_prefix_sha256"] != expected_prefix:
                raise ValueError("checkpoint data-window progress hash is inconsistent")
            restored_rng = checkpoint["rng_by_rank"][env.rank]

        if env.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[env.local_rank] if env.device_type == "cuda" else None,
            )
        raw_model = model.module if env.distributed else model
        validation_ids = dataset.fixed_validation_ids(
            args.eval_windows, args.validation_seed
        )
        if initial_validation_loss is None:
            evaluation = evaluate_full_validation if args.full_validation else evaluate_loss
            if args.full_validation:
                initial_validation_loss = evaluation(
                    model,
                    dataset,
                    batch_size=args.batch_size,
                    device=env.device,
                    device_type=env.device_type,
                    dtype=args.dtype,
                )
            else:
                initial_validation_loss = evaluation(
                    model,
                    dataset,
                    validation_ids,
                    batch_size=args.batch_size,
                    device=env.device,
                    device_type=env.device_type,
                    dtype=args.dtype,
                )
        if restored_rng is not None:
            restore_rng_state(restored_rng)

        target_update = args.max_updates or scheduler.updates_per_epoch
        if target_update > scheduler.updates_per_epoch:
            raise ValueError("C5 currently accepts at most one epoch")
        if target_update < next_update:
            raise ValueError("max-updates is behind the resumed checkpoint")

        invocation_start_update = next_update
        if env.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(env.device)
            torch.cuda.synchronize(env.device)
        started = time.perf_counter()
        for update in range(next_update, target_update):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro_step, window_ids in enumerate(scheduler.rank_window_ids(update)):
                is_last = micro_step + 1 == scheduler.local_micro_steps
                if env.distributed:
                    model.require_backward_grad_sync = is_last
                x, y = dataset.batch("train", window_ids, env.device)
                with autocast_context(env.device_type, args.dtype):
                    _, loss = model(x, y)
                    scaled_loss = loss / scheduler.local_micro_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                scaler.scale(scaled_loss).backward()
                micro_losses.append(loss.detach())
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            mean_loss = torch.stack(micro_losses).mean()
            if env.distributed:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= env.world_size
            loss_history.append(float(mean_loss))
            next_update = update + 1
            if env.master and (next_update == 1 or next_update % 10 == 0):
                print(
                    f"update {next_update}/{target_update} "
                    f"loss {loss_history[-1]:.6f}",
                    flush=True,
                )
            if args.checkpoint_every and next_update % args.checkpoint_every == 0:
                save_checkpoint(
                    path=Path(args.out_dir) / "checkpoint.pt",
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    next_update=next_update,
                    loss_history=loss_history,
                    initial_validation_loss=initial_validation_loss,
                    signature=signature,
                    scheduler=scheduler,
                    env=env,
                )

        if env.device_type == "cuda":
            torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - started
        if args.full_validation:
            final_validation_loss = evaluate_full_validation(
                model,
                dataset,
                batch_size=args.batch_size,
                device=env.device,
                device_type=env.device_type,
                dtype=args.dtype,
            )
        else:
            final_validation_loss = evaluate_loss(
                model,
                dataset,
                validation_ids,
                batch_size=args.batch_size,
                device=env.device,
                device_type=env.device_type,
                dtype=args.dtype,
            )
        checkpoint_path = Path(args.out_dir) / "checkpoint.pt"
        model_hash = save_checkpoint(
            path=checkpoint_path,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            next_update=next_update,
            loss_history=loss_history,
            initial_validation_loss=initial_validation_loss,
            signature=signature,
            scheduler=scheduler,
            env=env,
        )

        rank_metrics = gather_rank_objects(
            {
                "rank": env.rank,
                "elapsed_seconds_this_invocation": elapsed,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(env.device)
                    if env.device_type == "cuda" else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(env.device)
                    if env.device_type == "cuda" else None
                ),
            },
            env,
        )

        if env.master:
            plan = scheduler.epoch_metadata()
            complete = next_update == scheduler.updates_per_epoch
            slowest_elapsed = max(
                item["elapsed_seconds_this_invocation"] for item in rank_metrics
            )
            unique_covered = min(
                next_update * args.global_windows_per_update,
                dataset.train_window_count,
            )
            result = {
                "schema_version": 1,
                "stage": "C5",
                "backend": "native_ddp",
                "status": "complete" if complete else "partial",
                "configuration": vars(args),
                "checkpoint_signature": signature,
                "environment": environment_metadata(env),
                "git": git_metadata(),
                "dataset": {
                    "data_dir": str(dataset.data_dir),
                    "manifest_sha256": dataset.manifest_sha256,
                    "train_bin_sha256": sha256_file(dataset.train_path),
                    "val_bin_sha256": sha256_file(dataset.val_path),
                    "train_token_count": len(dataset._train),
                    "train_window_count": dataset.train_window_count,
                    "tail_token_count": (
                        (len(dataset._train) - 1) % args.block_size
                    ),
                    "validation_target_count": dataset.validation_target_count,
                    "validation_tail_token_count": dataset.validation_tail_token_count,
                },
                "epoch_plan": plan,
                "progress": {
                    "next_update": next_update,
                    "target_update": target_update,
                    "unique_windows_covered": unique_covered,
                    "coverage_ratio": unique_covered / dataset.train_window_count,
                    "scheduled_windows_processed": (
                        next_update * args.global_windows_per_update
                    ),
                    "padded_windows_in_complete_epoch": (
                        scheduler.padded_window_count if complete else None
                    ),
                    "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                },
                "metrics": {
                    "initial_validation_loss": initial_validation_loss,
                    "final_validation_loss": final_validation_loss,
                    "validation_scope": (
                        "all_target_tokens" if args.full_validation
                        else f"fixed_{args.eval_windows}_windows"
                    ),
                    "last_train_loss": loss_history[-1] if loss_history else None,
                    "elapsed_seconds_this_invocation": slowest_elapsed,
                    "per_rank": rank_metrics,
                    "peak_allocated_bytes": max(
                        (item["peak_allocated_bytes"] or 0) for item in rank_metrics
                    ) or None,
                    "peak_reserved_bytes": max(
                        (item["peak_reserved_bytes"] or 0) for item in rank_metrics
                    ) or None,
                },
                "artifacts": {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "model_state_sha256": model_hash,
                },
            }
            # Use the actual number of updates executed by this invocation.
            invocation_updates = target_update - invocation_start_update
            result["metrics"]["updates_this_invocation"] = invocation_updates
            result["metrics"]["tokens_per_second_this_invocation"] = (
                invocation_updates
                * args.global_windows_per_update
                * args.block_size
                / slowest_elapsed
                if slowest_elapsed > 0
                else math.nan
            )
            atomic_json_dump(result, Path(args.out_dir) / "result.json")
            print(
                f"saved {checkpoint_path} and {Path(args.out_dir) / 'result.json'}",
                flush=True,
            )
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
