"""Validated JSON scoring with separate base and memory-augmented calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.memory.normality_support import NormalitySupportResult
from src.memory.retriever import RetrievedMemory

from .backends import ChatBackend
from .score_utils import validate_anomaly_score


@dataclass(frozen=True)
class ScoringResult:
    anomaly_score: float
    visual_evidence: str
    temporal_evidence: str
    memory_comparison: str
    memory_used: bool
    evidence_conflict: bool
    raw_response: str
    normality_support: float

    def to_dict(self) -> dict:
        return {
            "anomaly_score": self.anomaly_score,
            "visual_evidence": self.visual_evidence,
            "temporal_evidence": self.temporal_evidence,
            "memory_comparison": self.memory_comparison,
            "memory_used": self.memory_used,
            "evidence_conflict": self.evidence_conflict,
            "normality_support": self.normality_support,
            "raw_response": self.raw_response,
        }


class StructuredAnomalyScorer:
    def __init__(
        self,
        backend: ChatBackend,
        *,
        prompt_dir: str | Path | None = None,
        max_retries: int = 2,
    ) -> None:
        self.backend = backend
        self.max_retries = max(0, max_retries)
        prompt_dir = Path(prompt_dir or Path(__file__).parents[1] / "prompts")
        self.system_template = (prompt_dir / "m3vad_system.txt").read_text(encoding="utf-8")
        self.user_template = (prompt_dir / "m3vad_user.txt").read_text(encoding="utf-8")
        self.call_count = 0

    def score(
        self,
        *,
        frame_caption: str,
        summary_10s: str,
        retrieved: Sequence[RetrievedMemory] = (),
        support: NormalitySupportResult | None = None,
        robust_memory: bool = False,
        base_only: bool = False,
    ) -> ScoringResult:
        support = support or NormalitySupportResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        memory_examples = [] if base_only else [item.to_prompt_dict() for item in retrieved]
        memory_description = (
            "Memory is disabled for this base-scoring call."
            if base_only
            else (
                "Trusted Memory passed additional deterministic reliability checks."
                if robust_memory
                else "Memory follows the legacy low-score admission rule."
            )
        )
        system_prompt = self.system_template.format(memory_description=memory_description)
        evidence_payload = {
            "frame_caption": frame_caption,
            "summary_10s": summary_10s,
            "normality_support": support.normality_support,
            "visual_similarity": support.visual_similarity,
            "caption_similarity": support.caption_similarity,
            "summary_similarity": support.summary_similarity,
            "memory_coverage": support.memory_coverage,
            "retrieved_normal_examples": memory_examples,
        }
        user_prompt = self.user_template.format(
            evidence_json=json.dumps(evidence_payload, indent=2, ensure_ascii=False)
        )

        last_error: Exception | None = None
        raw_response = ""
        for attempt in range(self.max_retries + 1):
            repair_instruction = ""
            if attempt:
                repair_instruction = (
                    "\nYour previous response was invalid. Return exactly one valid JSON object "
                    "matching the requested schema and use an allowed anomaly score."
                )
            self.call_count += 1
            raw_response = self.backend.generate(system_prompt, user_prompt + repair_instruction)
            try:
                return self._parse_result(
                    raw_response,
                    normality_support=support.normality_support,
                    memory_available=bool(memory_examples),
                )
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                last_error = exc
        raise ValueError(
            f"LLM returned invalid structured output after {self.max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _extract_json(response: str) -> Mapping[str, Any]:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("response JSON must be an object")
        return value

    @classmethod
    def _parse_result(
        cls,
        response: str,
        *,
        normality_support: float,
        memory_available: bool,
    ) -> ScoringResult:
        value = cls._extract_json(response)
        score = validate_anomaly_score(value.get("anomaly_score"))
        for field in ("visual_evidence", "temporal_evidence", "memory_comparison"):
            if field not in value or not isinstance(value[field], str):
                raise ValueError(f"missing or invalid field: {field}")
        memory_used = value.get("memory_used")
        conflict = value.get("evidence_conflict")
        if not isinstance(memory_used, bool) or not isinstance(conflict, bool):
            raise ValueError("memory_used and evidence_conflict must be booleans")
        if memory_used and not memory_available:
            raise ValueError("LLM claimed to use Memory when no examples were supplied")
        return ScoringResult(
            anomaly_score=score,
            visual_evidence=value["visual_evidence"],
            temporal_evidence=value["temporal_evidence"],
            memory_comparison=value["memory_comparison"],
            memory_used=memory_used,
            evidence_conflict=conflict,
            raw_response=response,
            normality_support=float(normality_support),
        )
