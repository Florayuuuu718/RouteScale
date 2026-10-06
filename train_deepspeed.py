"""DeepSpeed ZeRO entry point sharing RouteScale's deterministic C-stage data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from ddp_windows import GlobalWindowScheduler
from distributed_benchmark import (
    add_benchmark_arguments,
    run_distributed_benchmark,
)
from distributed_common import (
    MemmapTokenDataset,
    atomic_json_dump,
    autocast_context,
    barrier,
    build_adamw,
    build_model,
    destroy_distributed,
    environment_metadata,
    evaluate_loss,
    git_metadata,
    gather_rank_objects,
    initialize_distributed,
    seed_everything,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepspeed-config", default="config/deepspeed/zero0.json")
    parser.add_argument("--data-dir", default="data/tinystories_debug")
    parser.add_argument("--out-dir", default="results/c6_deepspeed/zero0_single_gpu")
    parser.add_argument("--max-updates", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-tag", default=None)
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
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use torch fused AdamW; enable in formal runs to match native DDP",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument(
        "--zero-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="construct ZeRO-3 parameters partitioned; enable for capacity tests",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="nccl")
    add_benchmark_arguments(
        parser, default_results_dir="results/c_backend_benchmark"
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace, local_accumulation: int,
                   world_size: int) -> dict:
    with Path(args.deepspeed_config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["train_micro_batch_size_per_gpu"] = args.batch_size
    config["gradient_accumulation_steps"] = local_accumulation
    config["train_batch_size"] = (
        args.batch_size * local_accumulation * world_size
    )
    config["gradient_clipping"] = args.grad_clip
    config["bf16"] = {"enabled": args.dtype == "bfloat16"}
    config["fp16"] = {"enabled": args.dtype == "float16"}
    config.setdefault("zero_optimization", {"stage": 0})
    return config


def signature(args, dataset, scheduler, stage: int, world_size: int) -> dict:
    return {
        "backend": "deepspeed",
        "zero_stage": stage,
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
        "fused_optimizer": args.fused_optimizer,
        "updates_per_epoch": scheduler.updates_per_epoch,
    }


def main() -> None:
    args = parse_args()
    if not args.benchmark and args.max_updates <= 0:
        raise ValueError("max-updates must be positive")
    if args.benchmark and (args.resume_dir or args.resume_tag):
        raise ValueError("benchmark mode cannot resume from a checkpoint")
    if args.benchmark and args.checkpoint_every:
        raise ValueError("benchmark mode cannot save periodic checkpoints")
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError(
            "DeepSpeed is not installed; run `uv sync --extra distributed`"
        ) from error

    env = initialize_distributed(device=args.device, backend=args.backend)
    try:
        if not env.distributed:
            raise RuntimeError(
                "launch DeepSpeed through torchrun, including for one GPU: "
                "uv run torchrun --standalone --nproc_per_node=1 train_deepspeed.py"
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
        resolved_config = resolve_config(
            args, scheduler.local_micro_steps, env.world_size
        )
        stage = int(resolved_config["zero_optimization"]["stage"])
        run_signature = signature(args, dataset, scheduler, stage, env.world_size)
        out_dir = Path(args.out_dir)
        if env.master:
            atomic_json_dump(resolved_config, out_dir / "resolved_deepspeed_config.json")
        barrier()

        init_context = (
            deepspeed.zero.Init(config_dict_or_path=resolved_config)
            if stage == 3 and args.zero_init
            else nullcontext()
        )
        with init_context:
            model = build_model(
                block_size=args.block_size,
                vocab_size=dataset.vocab_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
                bias=args.bias,
            )
        model_parameter_count = sum(
            getattr(parameter, "ds_numel", parameter.numel())
            for parameter in model.parameters()
        )
        optimizer = build_adamw(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            fused=args.fused_optimizer,
        )
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=resolved_config,
            dist_init_required=False,
        )

        def run_update(update: int, *, reduce_loss: bool) -> torch.Tensor:
            micro_losses = []
            starting_global_steps = engine.global_steps
            for window_ids in scheduler.rank_window_ids(update):
                x, y = dataset.batch("train", window_ids, env.device)
                with autocast_context(env.device_type, args.dtype):
                    _, loss = engine(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                engine.backward(loss)
                engine.step()
                micro_losses.append(loss.detach())
            if engine.global_steps != starting_global_steps + 1:
                raise RuntimeError(
                    "DeepSpeed did not produce exactly one optimizer update for "
                    "the configured accumulation window"
                )
            mean_loss = torch.stack(micro_losses).mean()
            if reduce_loss:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= env.world_size
            return mean_loss

        if args.benchmark:
            run_distributed_benchmark(
                args=args,
                env=env,
                dataset=dataset,
                scheduler=scheduler,
                backend="deepspeed",
                variant=f"deepspeed_zero{stage}",
                model_parameter_count=model_parameter_count,
                run_update=lambda update: run_update(update, reduce_loss=False),
                extra_configuration={
                    "zero_stage": stage,
                    "resolved_deepspeed_config": resolved_config,
                },
            )
            return

        next_update = 0
        loss_history: list[float] = []
        initial_validation_loss = None
        if args.resume_dir:
            load_path, client_state = engine.load_checkpoint(
                args.resume_dir,
                tag=args.resume_tag,
                load_optimizer_states=True,
                load_lr_scheduler_states=False,
            )
            if load_path is None:
                raise FileNotFoundError(f"no DeepSpeed checkpoint in {args.resume_dir}")
            if client_state["signature"] != run_signature:
                raise ValueError("DeepSpeed checkpoint signature mismatch")
            next_update = int(client_state["next_update"])
            loss_history = list(client_state["loss_history"])
            initial_validation_loss = float(client_state["initial_validation_loss"])
            if client_state["global_window_prefix_sha256"] != scheduler.hash_updates(0, next_update):
                raise ValueError("DeepSpeed checkpoint data-window progress mismatch")

        validation_ids = dataset.fixed_validation_ids(
            args.eval_windows, args.validation_seed
        )
        if initial_validation_loss is None:
            initial_validation_loss = evaluate_loss(
                engine,
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

        invocation_start = next_update
        torch.cuda.reset_peak_memory_stats(env.device)
        torch.cuda.synchronize(env.device)
        started = time.perf_counter()
        for update in range(next_update, args.max_updates):
            mean_loss = run_update(update, reduce_loss=True)
            loss_history.append(float(mean_loss))
            next_update = update + 1
            if env.master:
                print(
                    f"ZeRO-{stage} update {next_update}/{args.max_updates} "
                    f"loss {loss_history[-1]:.6f}",
                    flush=True,
                )
            if args.checkpoint_every and next_update % args.checkpoint_every == 0:
                engine.save_checkpoint(
                    str(out_dir / "checkpoints"),
                    tag=f"update-{next_update:08d}",
                    client_state={
                        "signature": run_signature,
                        "next_update": next_update,
                        "loss_history": loss_history,
                        "initial_validation_loss": initial_validation_loss,
                        "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
                    },
                    save_latest=True,
                )

        torch.cuda.synchronize(env.device)
        elapsed = time.perf_counter() - started
        final_validation_loss = evaluate_loss(
            engine,
            dataset,
            validation_ids,
            batch_size=args.batch_size,
            device=env.device,
            device_type=env.device_type,
            dtype=args.dtype,
        )
        checkpoint_tag = f"update-{next_update:08d}"
        engine.save_checkpoint(
            str(out_dir / "checkpoints"),
            tag=checkpoint_tag,
            client_state={
                "signature": run_signature,
                "next_update": next_update,
                "loss_history": loss_history,
                "initial_validation_loss": initial_validation_loss,
                "global_window_prefix_sha256": scheduler.hash_updates(0, next_update),
            },
            save_latest=True,
        )
        barrier()

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
                "stage": "C6",
                "backend": "deepspeed",
                "zero_stage": stage,
                "status": "smoke_complete",
                "configuration": vars(args),
                "resolved_deepspeed_config": resolved_config,
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
                "artifacts": {
                    "resolved_config": str(out_dir / "resolved_deepspeed_config.json"),
                    "checkpoint_dir": str(out_dir / "checkpoints"),
                    "checkpoint_tag": checkpoint_tag,
                },
            }
            atomic_json_dump(result, out_dir / "result.json")
            print(f"saved {out_dir / 'result.json'}", flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
