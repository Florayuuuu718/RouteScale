"""Allocation-only controls. Logical capacity is not serving throughput."""

from __future__ import annotations

from collections import deque
import math

from serving.h2_prefix_probe import prefix_disabled_manager


def static_window_manager(base, window):
    """Reserve a contiguous window, exposing ONLY live pages to the runner.

    Upstream prepare_decode uses block_table[-1] for writes. Exposing future
    reserved pages would silently write to the wrong page. Private reservations
    have real upstream block refcounts and are all released at request completion.
    """
    class StaticWindowManager(base):
        def __init__(self, num_blocks, block_size):
            super().__init__(num_blocks, block_size)
            if window % block_size:
                raise ValueError("Static window must be page aligned")
            self.width = window // block_size
            self.reservations = {}

        def first_fit(self):
            for start in range(len(self.blocks) - self.width + 1):
                if all(i not in self.used_block_ids for i in range(start, start + self.width)):
                    return list(range(start, start + self.width))
            return None

        def can_allocate(self, seq):
            return 0 if seq.num_blocks <= self.width and self.first_fit() is not None else -1

        def allocate(self, seq, num_cached_blocks):
            if num_cached_blocks or seq.block_table or self.can_allocate(seq) != 0:
                raise ValueError("Invalid static-window allocation")
            pages = self.first_fit()
            # Reorder the free queue, then use upstream allocation/refcount logic.
            free = set(self.free_block_ids)
            self.free_block_ids = deque(pages + sorted(free - set(pages)))
            self.reservations[seq.seq_id] = [self._allocate_block() for _ in pages]
            seq.block_table = pages[:seq.num_blocks]
            seq.num_cached_tokens = 0

        def can_append(self, seq):
            return seq.num_blocks <= self.width

        def may_append(self, seq):
            if not self.can_append(seq):
                raise ValueError("Request exceeds reserved window")
            seq.block_table[:] = self.reservations[seq.seq_id][:seq.num_blocks]

        def deallocate(self, seq):
            seq.block_table[:] = self.reservations.pop(seq.seq_id)
            super().deallocate(seq)

    return StaticWindowManager


class SpanAllocator:
    """CPU exact-token contiguous first-fit reference; no compaction or GPU kernel.

    Growth may relocate into another free span, allocating before freeing the
    old span. copied_tokens counts required logical KV copying, not measured I/O.
    """
    def __init__(self, budget, window):
        self.budget, self.window = budget, window
        self.live, self.spans = {}, {}
        self.copied_tokens = 0

    def holes(self):
        end, result = 0, []
        for start, size in sorted(self.spans.values()):
            if start > end:
                result.append((end, start - end))
            end = start + size
        if end < self.budget:
            result.append((end, self.budget - end))
        return result

    def allocate(self, key, tokens):
        if key in self.live or not 0 < tokens <= self.window:
            raise ValueError("Invalid request allocation")
        size = tokens
        hole = next((h for h in self.holes() if h[1] >= size), None)
        if hole is None:
            return False
        self.live[key], self.spans[key] = tokens, (hole[0], size)
        return True

    def release(self, key):
        del self.live[key], self.spans[key]

    def grow(self, key, tokens):
        if not self.live[key] <= tokens <= self.window:
            raise ValueError("Invalid growth")
        start, old = self.spans[key]
        delta = tokens - old
        adjacent = next((size for pos, size in self.holes() if pos == start + old), 0)
        if delta > adjacent:
            hole = next((h for h in self.holes() if h[1] >= tokens), None)
            if hole is None:
                return False
            start = hole[0]
            self.copied_tokens += self.live[key]
        self.live[key], self.spans[key] = tokens, (start, tokens)
        return True

    def snapshot(self):
        reserved = sum(size for _, size in self.spans.values())
        return dict(requests=len(self.live), live_tokens=sum(self.live.values()),
            reserved_tokens=reserved, free_tokens=self.budget-reserved,
            largest_free_span_tokens=max((size for _, size in self.holes()), default=0),
            copied_tokens=self.copied_tokens,
            page_tail_tokens=0, future_reserved_tokens=0)


class PageAllocator:
    """Replay actual pinned BlockManager/Sequence (prefix disabled), without GPU."""
    def __init__(self, base, sequence, blocks, block_size, window, *, static=False):
        cls = static_window_manager(base, window) if static else prefix_disabled_manager(base)
        self.manager, self.sequence = cls(blocks, block_size), sequence
        self.block_size, self.window, self.budget = block_size, window, blocks * block_size
        self.live = {}

    def allocate(self, key, tokens):
        if key in self.live or not 0 < tokens <= self.window:
            raise ValueError("Invalid request allocation")
        seq = self.sequence([key + 1] * tokens)
        cached = self.manager.can_allocate(seq)
        if cached < 0:
            return False
        self.manager.allocate(seq, cached)
        self.live[key] = seq
        return True

    def release(self, key):
        self.manager.deallocate(self.live.pop(key))

    def grow(self, key, tokens):
        seq = self.live[key]
        if not len(seq) <= tokens <= self.window:
            raise ValueError("Invalid growth")
        needed = math.ceil(tokens/self.block_size)-len(seq.block_table)
        if not hasattr(self.manager, "reservations") and needed > len(self.manager.free_block_ids):
            return False
        while len(seq) < tokens:
            seq.append_token(key + 1)
            if not self.manager.can_append(seq):
                raise AssertionError("Growth precheck disagrees with upstream")
            self.manager.may_append(seq)
        return True

    def snapshot(self):
        live = sum(len(s) for s in self.live.values())
        used = len(self.manager.used_block_ids) * self.block_size
        is_static = hasattr(self.manager, "reservations")
        return dict(requests=len(self.live), live_tokens=live, reserved_tokens=used,
            free_tokens=self.budget-used,
            page_tail_tokens=sum((-len(s)) % self.block_size for s in self.live.values()),
            future_reserved_tokens=sum(self.window-s.num_blocks*self.block_size for s in self.live.values()) if is_static else 0,
            copied_tokens=0)


def assert_conservation(allocator):
    state = allocator.snapshot()
    if (state["reserved_tokens"] + state["free_tokens"] != allocator.budget
            or state["reserved_tokens"] != state["live_tokens"] + state["page_tail_tokens"] + state["future_reserved_tokens"]):
        raise AssertionError("KV budget partition mismatch")
    if isinstance(allocator, PageAllocator):
        manager = allocator.manager
        free, used = set(manager.free_block_ids), manager.used_block_ids
        if free & used or free | used != set(range(len(manager.blocks))) or len(free) != len(manager.free_block_ids):
            raise AssertionError("Page partition mismatch")
        pages = [p for s in allocator.live.values() for p in
            (manager.reservations[s.seq_id] if hasattr(manager, "reservations") else s.block_table)]
        if len(pages) != len(set(pages)) or set(pages) != used:
            raise AssertionError("Page ownership mismatch")
        if any(b.ref_count != int(b.block_id in used) for b in manager.blocks):
            raise AssertionError("Page refcount mismatch")
    return state


def allocator_report(config, base, sequence):
    block, blocks, window = [config[k] for k in ("block_size", "num_blocks", "max_model_len")]
    budget = block * blocks
    factories = {
        "static_window": lambda: PageAllocator(base, sequence, blocks, block, window, static=True),
        "contiguous_exact": lambda: SpanAllocator(budget, window),
        "paged": lambda: PageAllocator(base, sequence, blocks, block, window),
    }
    capacity = []
    for tokens in config["capacity_lengths"]:
        for policy, factory in factories.items():
            allocator, key = factory(), 0
            while allocator.allocate(key, tokens):
                assert_conservation(allocator)
                key += 1
            full = assert_conservation(allocator)
            if key != budget // (window if policy == "static_window" else
                    math.ceil(tokens/block)*block if policy == "paged" else tokens):
                raise AssertionError("Homogeneous capacity disagrees with byte accounting")
            for index in list(allocator.live):
                allocator.release(index)
                assert_conservation(allocator)
            capacity.append(dict(policy=policy, live_kv_tokens_per_request=tokens,
                at_capacity=full, after_release=assert_conservation(allocator)))
    spec, fragmentation = config["fragmentation"], {}
    # Static-window cannot admit this initial population; do not compare a
    # different, already-blocked initial state in the hole-reuse experiment.
    for policy in ("contiguous_exact", "paged"):
        allocator = factories[policy]()
        for key in range(spec["initial_requests"]):
            if not allocator.allocate(key, spec["initial_tokens"]):
                raise AssertionError("Fragmentation initial population must fit on both sides")
            assert_conservation(allocator)
        for key in range(0, spec["initial_requests"], spec["release_stride"]):
            allocator.release(key)
            assert_conservation(allocator)
        before = assert_conservation(allocator)
        accepted = allocator.allocate(100, spec["new_tokens"])
        tables = {str(k): list(s.block_table) for k, s in allocator.live.items()} if policy == "paged" else None
        fragmentation[policy] = dict(before=before, new_tokens=spec["new_tokens"], accepted=accepted,
            after=assert_conservation(allocator), page_tables=tables)
        for key in list(allocator.live):
            allocator.release(key)
        fragmentation[policy]["after_release"] = assert_conservation(allocator)
    growth = {}
    for policy, factory in factories.items():
        allocator = factory()
        for key in range(4):
            if not allocator.allocate(key, 256):
                raise AssertionError("Growth initial population cannot fit")
        before = assert_conservation(allocator)
        accepted = allocator.grow(0, 257)
        growth[policy] = dict(before=before, accepted=accepted, after=assert_conservation(allocator))
        for key in list(allocator.live):
            allocator.release(key)
        growth[policy]["after_release"] = assert_conservation(allocator)
    return dict(evidence_class="cpu_allocator_replay", capacity=capacity,
        fragmentation=fragmentation, growth=growth,
        limitations="No model execution, physical GPU allocation, throughput or latency. Contiguous exact-token allocator is an explicit reference, not upstream code. Static windows are reservation, not external fragmentation.")
