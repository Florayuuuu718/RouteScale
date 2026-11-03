from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args
    from .workers import WorkerGroup
    from minisgl.env import ENV

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    # This repair series relies on a single ordered request/abort producer and
    # no in-flight overlap results. Other modes require their own acceptance.
    if server_args.num_tokenizer != 0:
        raise ValueError("Lifecycle patches require the shared ordered tokenizer")
    if min(server_args.request_timeout_s, server_args.startup_timeout_s) <= 0:
        raise ValueError("Lifecycle timeouts must be positive")
    if server_args.max_total_requests < 1:
        raise ValueError("max_total_requests must be positive")
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> WorkerGroup:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()
        workers = WorkerGroup()

        try:
            for i in range(world_size):
                new_args = replace(
                    server_args,
                    tp_info=DistributedInfo(i, world_size),
                )
                workers.start(
                    target=_run_scheduler,
                    args=(new_args, ack_queue),
                    daemon=False,
                    name=f"minisgl-TP{i}-scheduler",
                )

            num_tokenizers = server_args.num_tokenizer
            # DeTokenizer, only 1
            workers.start(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_detokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": num_tokenizers,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name="minisgl-detokenizer-0",
            )
            for i in range(num_tokenizers):
                workers.start(
                    target=tokenize_worker,
                    kwargs={
                        "tokenizer_path": server_args.model_path,
                        "addr": server_args.zmq_tokenizer_addr,
                        "backend_addr": server_args.zmq_backend_addr,
                        "frontend_addr": server_args.zmq_frontend_addr,
                        "local_bs": 1,
                        "create": server_args.tokenizer_create_addr,
                        "tokenizer_id": i,
                        "ack_queue": ack_queue,
                    },
                    daemon=False,
                    name=f"minisgl-tokenizer-{i}",
                )

            # Wait for acknowledgments from all worker processes:
            # - world_size schedulers (but only primary rank sends ack)
            # - num_tokenizers tokenizers
            # - 1 detokenizer
            # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
            workers.wait_ready(ack_queue, num_tokenizers + 2, server_args.startup_timeout_s)
            for message in workers.ready_messages:
                logger.info(message)
        except BaseException:
            workers.stop()
            raise
        finally:
            ack_queue.close()
            ack_queue.join_thread()
        return workers

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
