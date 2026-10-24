"""Memory admission policies for legacy and contamination-resistant modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .memory_entry import MemoryEntry, MemoryStatus


@dataclass(frozen=True)
class UpdateDecision:
    accepted: bool
    status: MemoryStatus
    reasons: List[str] = field(default_factory=list)


class LegacyMemoryPolicy:
    def __init__(self, score_threshold: float = 0.2) -> None:
        self.score_threshold = score_threshold

    def evaluate(self, entry: MemoryEntry, *, is_duplicate: bool = False) -> UpdateDecision:
        if entry.final_anomaly_score <= self.score_threshold:
            return UpdateDecision(True, MemoryStatus.TRUSTED, ["legacy_score_gate_passed"])
        return UpdateDecision(False, MemoryStatus.REJECTED, ["final_score_above_threshold"])


class RobustMemoryPolicy:
    def __init__(
        self,
        *,
        score_threshold: float = 0.2,
        alignment_threshold: float = 0.45,
        temporal_stability_threshold: float = 0.60,
        overall_reliability_threshold: float = 0.60,
    ) -> None:
        self.score_threshold = score_threshold
        self.alignment_threshold = alignment_threshold
        self.temporal_stability_threshold = temporal_stability_threshold
        self.overall_reliability_threshold = overall_reliability_threshold

    def evaluate(self, entry: MemoryEntry, *, is_duplicate: bool = False) -> UpdateDecision:
        reasons: List[str] = []
        metrics = entry.reliability_metrics
        if entry.final_anomaly_score > self.score_threshold:
            reasons.append("final_score_above_threshold")
        if entry.base_anomaly_score > self.score_threshold:
            reasons.append("base_score_above_threshold")
        alignment = min(
            float(metrics.get("visual_caption_alignment", 0.0)),
            float(metrics.get("visual_summary_alignment", 0.0)),
        )
        if alignment < self.alignment_threshold:
            reasons.append("visual_text_alignment_too_low")
        if float(metrics.get("temporal_stability", 0.0)) < self.temporal_stability_threshold:
            reasons.append("local_scores_unstable")
        if bool(metrics.get("evidence_conflict", False)):
            reasons.append("multimodal_evidence_conflict")
        if float(metrics.get("overall_reliability", 0.0)) < self.overall_reliability_threshold:
            reasons.append("overall_reliability_too_low")
        if is_duplicate:
            reasons.append("duplicate_normal_pattern")

        if reasons:
            return UpdateDecision(False, MemoryStatus.PENDING, reasons)
        return UpdateDecision(True, MemoryStatus.TRUSTED, ["robust_reliability_gate_passed"])
