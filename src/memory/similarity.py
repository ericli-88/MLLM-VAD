"""Similarity primitives that never compare vectors from different encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"embedding dimensions differ: {len(left)} != {len(right)}")
    if not left:
        return 0.0
    left_n = l2_normalize(left)
    right_n = l2_normalize(right)
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left_n, right_n))))


def cosine_to_support(similarity: float) -> float:
    """Map cosine range [-1, 1] to evidence strength [0, 1].

    This is a bounded support transform, not probability calibration.
    """

    return max(0.0, min(1.0, (float(similarity) + 1.0) / 2.0))


@dataclass(frozen=True)
class SimilarityWeights:
    visual: float = 0.45
    caption: float = 0.20
    summary: float = 0.35

    def normalized(self) -> "SimilarityWeights":
        values = (self.visual, self.caption, self.summary)
        if any(value < 0 for value in values):
            raise ValueError("similarity weights must be non-negative")
        total = sum(values)
        if total <= 0:
            raise ValueError("at least one similarity weight must be positive")
        return SimilarityWeights(*(value / total for value in values))


def fused_similarity(
    visual: float,
    caption: float,
    summary: float,
    weights: SimilarityWeights,
) -> float:
    weights = weights.normalized()
    return (
        weights.visual * float(visual)
        + weights.caption * float(caption)
        + weights.summary * float(summary)
    )


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default
