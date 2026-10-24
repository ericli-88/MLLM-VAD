"""Two-stage, diversity-aware retrieval over trusted normality memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .memory_entry import MemoryEntry, MemoryStatus
from .similarity import SimilarityWeights, cosine_similarity, fused_similarity


@dataclass(frozen=True)
class QueryEmbeddings:
    visual: Sequence[float]
    caption: Sequence[float]
    summary: Sequence[float]


@dataclass(frozen=True)
class RetrievedMemory:
    entry: MemoryEntry
    visual_similarity: float
    caption_similarity: float
    summary_similarity: float
    combined_similarity: float

    def to_prompt_dict(self) -> dict:
        return {
            "memory_id": self.entry.memory_id,
            "frame_index": self.entry.frame_index,
            "timestamp": self.entry.timestamp,
            "frame_caption": self.entry.frame_caption,
            "summary_10s": self.entry.summary_10s,
            "visual_similarity": round(self.visual_similarity, 6),
            "caption_similarity": round(self.caption_similarity, 6),
            "summary_similarity": round(self.summary_similarity, 6),
            "combined_similarity": round(self.combined_similarity, 6),
        }


class MemoryRetriever:
    """Scene-first retrieval followed by action/summary re-ranking.

    Visual similarity forms the scene-aware candidate pool. Caption and temporal
    summary similarity then distinguish actions that occur in similar scenes.
    All comparisons stay within their own encoder space.
    """

    def __init__(
        self,
        *,
        top_k: int = 3,
        stage1_multiplier: int = 4,
        weights: SimilarityWeights | None = None,
        min_temporal_gap_frames: int = 150,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.stage1_multiplier = max(1, stage1_multiplier)
        self.weights = (weights or SimilarityWeights()).normalized()
        self.min_temporal_gap_frames = max(0, min_temporal_gap_frames)

    def retrieve(
        self,
        entries: Iterable[MemoryEntry],
        query: QueryEmbeddings,
        *,
        video_id: str,
        query_frame_index: int,
        exclude_memory_id: Optional[str] = None,
    ) -> List[RetrievedMemory]:
        eligible = [
            entry
            for entry in entries
            if entry.status == MemoryStatus.TRUSTED
            and entry.video_id == video_id
            and entry.frame_index < query_frame_index
            and entry.memory_id != exclude_memory_id
        ]
        if not eligible:
            return []

        if self.weights.visual > 0:
            # Multimodal mode: use the visual representation as a scene-aware gate.
            stage1_score = lambda item: cosine_similarity(query.visual, item.visual_embedding)
        else:
            # Text-only ablation: do not leak visual evidence into candidate selection.
            stage1_score = lambda item: (
                self.weights.caption
                * cosine_similarity(query.caption, item.caption_embedding)
                + self.weights.summary
                * cosine_similarity(query.summary, item.summary_embedding)
            )
        scene_ranked = sorted(eligible, key=stage1_score, reverse=True)
        candidate_count = min(len(scene_ranked), self.top_k * self.stage1_multiplier)
        candidates = scene_ranked[:candidate_count]

        reranked: List[RetrievedMemory] = []
        for entry in candidates:
            visual = cosine_similarity(query.visual, entry.visual_embedding)
            caption = cosine_similarity(query.caption, entry.caption_embedding)
            summary = cosine_similarity(query.summary, entry.summary_embedding)
            reranked.append(
                RetrievedMemory(
                    entry=entry,
                    visual_similarity=visual,
                    caption_similarity=caption,
                    summary_similarity=summary,
                    combined_similarity=fused_similarity(
                        visual, caption, summary, self.weights
                    ),
                )
            )
        reranked.sort(key=lambda item: item.combined_similarity, reverse=True)

        selected: List[RetrievedMemory] = []
        for item in reranked:
            if all(
                abs(item.entry.frame_index - chosen.entry.frame_index)
                >= self.min_temporal_gap_frames
                for chosen in selected
            ):
                selected.append(item)
            if len(selected) == self.top_k:
                break
        return selected
