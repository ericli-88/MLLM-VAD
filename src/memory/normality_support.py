"""Deterministic evidence support from retrieved normal examples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .retriever import RetrievedMemory
from .similarity import cosine_to_support, mean


@dataclass(frozen=True)
class NormalitySupportResult:
    normality_support: float
    visual_similarity: float
    caption_similarity: float
    summary_similarity: float
    memory_coverage: float
    modality_agreement: float
    reference_reliability: float

    def to_dict(self) -> dict:
        return {
            key: round(float(value), 6)
            for key, value in self.__dict__.items()
        }


class NormalitySupport:
    """Compute bounded support evidence; the result is not P(normal)."""

    def __init__(self, expected_k: int = 3) -> None:
        self.expected_k = max(1, expected_k)

    def compute(
        self,
        retrieved: Iterable[RetrievedMemory],
        *,
        query_reliability: float = 1.0,
    ) -> NormalitySupportResult:
        items = list(retrieved)
        if not items:
            return NormalitySupportResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        visual = mean(cosine_to_support(item.visual_similarity) for item in items)
        caption = mean(cosine_to_support(item.caption_similarity) for item in items)
        summary = mean(cosine_to_support(item.summary_similarity) for item in items)
        combined = mean(cosine_to_support(item.combined_similarity) for item in items)
        modality_values = [visual, caption, summary]
        modality_mean = mean(modality_values)
        variance = mean((value - modality_mean) ** 2 for value in modality_values)
        agreement = max(0.0, 1.0 - 2.0 * math.sqrt(variance))
        coverage = min(1.0, len(items) / self.expected_k)
        reference_reliability = mean(
            float(item.entry.reliability_metrics.get("overall_reliability", 1.0))
            for item in items
        )
        reliability = max(
            0.0,
            min(1.0, 0.5 * reference_reliability + 0.5 * float(query_reliability)),
        )
        support = (
            0.45 * combined
            + 0.20 * agreement
            + 0.20 * coverage
            + 0.15 * reliability
        )
        return NormalitySupportResult(
            normality_support=max(0.0, min(1.0, support)),
            visual_similarity=visual,
            caption_similarity=caption,
            summary_similarity=summary,
            memory_coverage=coverage,
            modality_agreement=agreement,
            reference_reliability=reliability,
        )
