"""Typed metadata stored for one inferred-normal target frame."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping


ALLOWED_SCORES = tuple(round(index / 10, 1) for index in range(11))


class MemoryStatus(str, Enum):
    PENDING = "pending"
    TRUSTED = "trusted"
    REJECTED = "rejected"


def make_memory_id(video_id: str, frame_index: int) -> str:
    """Create a stable ID so retries cannot insert the same frame twice."""

    value = f"{video_id}:{int(frame_index)}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


@dataclass
class MemoryEntry:
    memory_id: str
    video_id: str
    frame_index: int
    timestamp: float
    frame_caption: str
    summary_10s: str
    base_anomaly_score: float
    final_anomaly_score: float
    visual_embedding: List[float]
    caption_embedding: List[float]
    summary_embedding: List[float]
    reliability_metrics: Dict[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.PENDING

    def __post_init__(self) -> None:
        if not self.memory_id:
            self.memory_id = make_memory_id(self.video_id, self.frame_index)
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        for name, score in (
            ("base_anomaly_score", self.base_anomaly_score),
            ("final_anomaly_score", self.final_anomaly_score),
        ):
            if round(float(score), 1) not in ALLOWED_SCORES:
                raise ValueError(f"{name} must be one of {ALLOWED_SCORES}; got {score}")
        self.base_anomaly_score = round(float(self.base_anomaly_score), 1)
        self.final_anomaly_score = round(float(self.final_anomaly_score), 1)
        self.visual_embedding = [float(value) for value in self.visual_embedding]
        self.caption_embedding = [float(value) for value in self.caption_embedding]
        self.summary_embedding = [float(value) for value in self.summary_embedding]
        if isinstance(self.status, str):
            self.status = MemoryStatus(self.status)

    @classmethod
    def create(
        cls,
        *,
        video_id: str,
        frame_index: int,
        timestamp: float,
        frame_caption: str,
        summary_10s: str,
        base_anomaly_score: float,
        final_anomaly_score: float,
        visual_embedding: List[float],
        caption_embedding: List[float],
        summary_embedding: List[float],
        reliability_metrics: Mapping[str, Any] | None = None,
        status: MemoryStatus = MemoryStatus.PENDING,
    ) -> "MemoryEntry":
        return cls(
            memory_id=make_memory_id(video_id, frame_index),
            video_id=video_id,
            frame_index=frame_index,
            timestamp=timestamp,
            frame_caption=frame_caption,
            summary_10s=summary_10s,
            base_anomaly_score=base_anomaly_score,
            final_anomaly_score=final_anomaly_score,
            visual_embedding=visual_embedding,
            caption_embedding=caption_embedding,
            summary_embedding=summary_embedding,
            reliability_metrics=dict(reliability_metrics or {}),
            status=status,
        )

    def to_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        if not include_embeddings:
            data.pop("visual_embedding", None)
            data.pop("caption_embedding", None)
            data.pop("summary_embedding", None)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryEntry":
        return cls(**dict(data))
