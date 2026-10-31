"""Native RouteScale incremental inference."""

from serving.native.generation import (
    GenerationResult,
    greedy_generate_cached,
    greedy_generate_naive,
)

__all__ = [
    "GenerationResult",
    "greedy_generate_cached",
    "greedy_generate_naive",
]

