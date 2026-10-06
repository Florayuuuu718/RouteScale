"""Deterministic global-window planning for RouteScale DDP experiments.

The scheduler creates one world-size-independent global sequence, then maps
each update to rank, micro-step, and batch-slot coordinates.  Window IDs refer
to non-overlapping training windows; byte/token offsets are derived by
multiplying an ID by ``block_size``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Literal

import numpy as np


TailPolicy = Literal["drop", "pad"]


def hash_window_ids(window_ids: np.ndarray) -> str:
    """Return a platform-independent SHA256 for an ordered window-ID array."""

    values = np.asarray(window_ids, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    complete_window_count: int
    scheduled_window_count: int
    updates_per_epoch: int
    dropped_window_count: int
    padded_window_count: int
    tail_policy: TailPolicy


class GlobalWindowScheduler:
    """Build and shard deterministic, shuffled epochs of global windows.

    The global order depends only on ``window_count``, ``seed``, ``epoch`` and
    the global update size.  It does not depend on world size.  For a fixed
    global update size, 1/2/4-rank strong-scaling runs therefore consume the
    same ordered windows and differ only in their rank assignment.
    """

    def __init__(
        self,
        *,
        window_count: int,
        global_windows_per_update: int,
        world_size: int,
        rank: int,
        batch_size: int,
        seed: int,
        tail_policy: TailPolicy = "drop",
    ) -> None:
        if window_count <= 0:
            raise ValueError("window_count must be positive")
        if global_windows_per_update <= 0:
            raise ValueError("global_windows_per_update must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if tail_policy not in ("drop", "pad"):
            raise ValueError("tail_policy must be 'drop' or 'pad'")
        if global_windows_per_update % world_size:
            raise ValueError(
                "global_windows_per_update must be divisible by world_size"
            )

        local_windows = global_windows_per_update // world_size
        if local_windows % batch_size:
            raise ValueError(
                "windows per rank and update must be divisible by batch_size"
            )

        remainder = window_count % global_windows_per_update
        if tail_policy == "drop":
            scheduled_count = window_count - remainder
            dropped_count = remainder
            padded_count = 0
        else:
            padded_count = (
                global_windows_per_update - remainder
            ) % global_windows_per_update
            scheduled_count = window_count + padded_count
            dropped_count = 0
        if scheduled_count == 0:
            raise ValueError(
                "tail_policy='drop' leaves no complete global update; use 'pad'"
            )

        self.window_count = window_count
        self.global_windows_per_update = global_windows_per_update
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.seed = seed
        self.tail_policy = tail_policy
        self.local_windows_per_update = local_windows
        self.local_micro_steps = local_windows // batch_size
        self.scheduled_window_count = scheduled_count
        self.updates_per_epoch = scheduled_count // global_windows_per_update
        self.dropped_window_count = dropped_count
        self.padded_window_count = padded_count
        self._cached_epoch: int | None = None
        self._cached_original_order: np.ndarray | None = None
        self._cached_scheduled_order: np.ndarray | None = None

    def _orders_for_epoch(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self._cached_epoch == epoch:
            assert self._cached_original_order is not None
            assert self._cached_scheduled_order is not None
            return self._cached_original_order, self._cached_scheduled_order

        seed_sequence = np.random.SeedSequence([self.seed, epoch])
        generator = np.random.Generator(np.random.PCG64(seed_sequence))
        original = generator.permutation(self.window_count).astype(
            np.int64, copy=False
        )
        if self.tail_policy == "drop":
            scheduled = original[: self.scheduled_window_count]
        elif self.padded_window_count:
            padding = np.resize(original, self.padded_window_count)
            scheduled = np.concatenate((original, padding))
        else:
            scheduled = original

        self._cached_epoch = epoch
        self._cached_original_order = original
        self._cached_scheduled_order = scheduled
        return original, scheduled

    def epoch_plan(self, epoch: int = 0) -> EpochPlan:
        return EpochPlan(
            epoch=epoch,
            complete_window_count=self.window_count,
            scheduled_window_count=self.scheduled_window_count,
            updates_per_epoch=self.updates_per_epoch,
            dropped_window_count=self.dropped_window_count,
            padded_window_count=self.padded_window_count,
            tail_policy=self.tail_policy,
        )

    def epoch_metadata(self, epoch: int = 0) -> dict[str, int | str]:
        return asdict(self.epoch_plan(epoch))

    def epoch_window_ids(self, epoch: int = 0) -> np.ndarray:
        """Return the scheduled epoch, including padding and excluding drops."""

        _, scheduled = self._orders_for_epoch(epoch)
        return scheduled.copy()

    def dropped_window_ids(self, epoch: int = 0) -> np.ndarray:
        """Return the shuffled tail omitted by ``tail_policy='drop'``."""

        original, _ = self._orders_for_epoch(epoch)
        if not self.dropped_window_count:
            return np.empty(0, dtype=np.int64)
        return original[-self.dropped_window_count :].copy()

    def padded_window_ids(self, epoch: int = 0) -> np.ndarray:
        """Return repeated IDs appended by ``tail_policy='pad'``."""

        _, scheduled = self._orders_for_epoch(epoch)
        if not self.padded_window_count:
            return np.empty(0, dtype=np.int64)
        return scheduled[-self.padded_window_count :].copy()

    def global_window_ids(self, update: int) -> np.ndarray:
        """Return one update's ordered global window IDs."""

        if update < 0:
            raise ValueError("update must be non-negative")
        epoch, update_in_epoch = divmod(update, self.updates_per_epoch)
        _, scheduled = self._orders_for_epoch(epoch)
        start = update_in_epoch * self.global_windows_per_update
        stop = start + self.global_windows_per_update
        return scheduled[start:stop].copy()

    def rank_window_ids(self, update: int, rank: int | None = None) -> np.ndarray:
        """Return ``[local_micro_steps, batch_size]`` IDs for one rank."""

        selected_rank = self.rank if rank is None else rank
        if not 0 <= selected_rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        global_ids = self.global_window_ids(update)
        rank_major = global_ids.reshape(
            self.world_size, self.local_windows_per_update
        )
        return rank_major[selected_rank].reshape(
            self.local_micro_steps, self.batch_size
        ).copy()

    def rank_window_offsets(
        self, update: int, block_size: int, rank: int | None = None
    ) -> np.ndarray:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        return self.rank_window_ids(update, rank) * block_size

    def hash_updates(
        self, start_update: int, update_count: int, rank: int | None = None
    ) -> str:
        """Hash an ordered range of global or rank-local update windows."""

        if start_update < 0 or update_count < 0:
            raise ValueError("update range must be non-negative")
        digest = hashlib.sha256()
        for update in range(start_update, start_update + update_count):
            values = (
                self.global_window_ids(update)
                if rank is None
                else self.rank_window_ids(update, rank).reshape(-1)
            )
            digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
        return digest.hexdigest()
