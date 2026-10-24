"""Configuration schema for reproducible M3VAD ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from src.memory.similarity import SimilarityWeights


class ExperimentMode(str, Enum):
    A_BASELINE = "A"
    B_ONEVISION = "B"
    C_LEGACY_TEXT_MEMORY = "C"
    D_MULTIMODAL_MEMORY = "D"
    E_ROBUST_MEMORY = "E"


@dataclass
class OneVisionConfig:
    enabled: bool = True
    model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    num_samples: int = 16
    duration_seconds: float = 10.0
    load_in_4bit: bool = False
    max_new_tokens: int = 256


@dataclass
class MemoryConfig:
    enabled: bool = True
    mode: str = "robust"
    retrieval: str = "multimodal"
    capacity: int = 500
    top_k: int = 3
    stage1_multiplier: int = 4
    min_temporal_gap_frames: int = 150
    duplicate_threshold: float = 0.985
    score_threshold: float = 0.2
    alignment_threshold: float = 0.45
    temporal_stability_threshold: float = 0.60
    overall_reliability_threshold: float = 0.60
    weights: SimilarityWeights = field(default_factory=SimilarityWeights)


@dataclass
class M3VADConfig:
    experiment: ExperimentMode = ExperimentMode.E_ROBUST_MEMORY
    target_frame_interval: int = 10
    default_fps: float = 30.0
    fps_tolerance: float = 0.5
    imagefile_template: str = "{:06d}.jpg"
    onevision: OneVisionConfig = field(default_factory=OneVisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        if self.target_frame_interval <= 0:
            raise ValueError("target_frame_interval must be positive")
        if self.default_fps <= 0:
            raise ValueError("default_fps must be positive")
        if self.memory.mode not in {"off", "legacy", "robust"}:
            raise ValueError("memory.mode must be off, legacy, or robust")
        if self.memory.retrieval not in {"text", "multimodal"}:
            raise ValueError("memory.retrieval must be text or multimodal")
        self.memory.weights.normalized()
        if self.experiment == ExperimentMode.A_BASELINE and (
            self.onevision.enabled or self.memory.enabled
        ):
            raise ValueError("Experiment A must disable OneVision and Memory")
        if self.experiment == ExperimentMode.B_ONEVISION and self.memory.enabled:
            raise ValueError("Experiment B must disable Memory")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M3VADConfig":
        onevision = OneVisionConfig(**data.get("onevision", {}))
        memory_data = dict(data.get("memory", {}))
        weight_data = memory_data.pop("weights", {})
        memory = MemoryConfig(
            **memory_data,
            weights=SimilarityWeights(**weight_data),
        )
        config = cls(
            experiment=ExperimentMode(data.get("experiment", "E")),
            target_frame_interval=int(data.get("target_frame_interval", 10)),
            default_fps=float(data.get("default_fps", 30.0)),
            fps_tolerance=float(data.get("fps_tolerance", 0.5)),
            imagefile_template=data.get("imagefile_template", "{:06d}.jpg"),
            onevision=onevision,
            memory=memory,
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "M3VADConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
