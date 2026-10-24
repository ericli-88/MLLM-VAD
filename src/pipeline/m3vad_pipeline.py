"""Offline, chronological M3VAD inference with video-isolated Dynamic Memory."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from src.config import ExperimentMode, M3VADConfig
from src.data.video_record import VideoRecord
from src.memory.memory_bank import DynamicMemoryBank
from src.memory.memory_entry import MemoryEntry
from src.memory.multimodal_encoder import (
    HashMultimodalEncoder,
    ImageBindVisualEncoder,
    MultimodalEncoder,
    SentenceTransformerTextEncoder,
)
from src.memory.normality_support import NormalitySupport
from src.memory.retriever import MemoryRetriever, QueryEmbeddings
from src.memory.update_policy import LegacyMemoryPolicy, RobustMemoryPolicy
from src.models.llava_onevision_summarizer import (
    LlavaOneVisionSummarizer,
    TemporalWindow,
    build_centered_window,
)
from src.scoring.backends import (
    HeuristicMockBackend,
    Llama2ChatBackend,
    TransformersChatBackend,
)
from src.scoring.score_utils import interpolate_target_scores
from src.scoring.structured_scorer import StructuredAnomalyScorer


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_fps(video_id: str, video_dir: Path | None, fallback_fps: float) -> float:
    if video_dir is None:
        return fallback_fps
    candidates = [
        path
        for suffix in (".mp4", ".avi", ".mov", ".mkv")
        for path in [video_dir / f"{video_id}{suffix}"]
        if path.exists()
    ]
    if not candidates:
        warnings.warn(
            f"No source video found for {video_id}; using configured {fallback_fps} FPS.",
            RuntimeWarning,
        )
        return fallback_fps
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV unavailable; using configured FPS.", RuntimeWarning)
        return fallback_fps
    capture = cv2.VideoCapture(str(candidates[0]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return fps if fps > 0 else fallback_fps


def _load_caption(captions: Mapping[str, object], frame_index: int) -> str:
    value = captions.get(str(frame_index))
    if value is None:
        raise KeyError(
            f"BLIP-2 caption missing for target frame {frame_index}. "
            "Generate captions with --frame_interval 10."
        )
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values()).strip()
    raise TypeError(f"unsupported caption value for frame {frame_index}: {type(value)}")


def _temporal_stability(previous_scores: Sequence[float], current_score: float) -> float:
    values = [*previous_scores[-3:], float(current_score)]
    if len(values) <= 1:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (max(values) - min(values))))


class M3VADPipeline:
    def __init__(
        self,
        *,
        config: M3VADConfig,
        captions_dir: str | Path,
        output_dir: str | Path,
        scorer: StructuredAnomalyScorer,
        encoder: MultimodalEncoder,
        onevision_summarizer: LlavaOneVisionSummarizer | None,
        summaries_dir: str | Path | None = None,
        video_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.captions_dir = Path(captions_dir)
        self.output_dir = Path(output_dir)
        self.scorer = scorer
        self.encoder = encoder
        self.onevision_summarizer = onevision_summarizer
        self.summaries_dir = Path(summaries_dir) if summaries_dir else None
        self.video_dir = Path(video_dir) if video_dir else None
        self.support_calculator = NormalitySupport(config.memory.top_k)

    def process_video(self, video: VideoRecord) -> dict:
        if self.config.experiment == ExperimentMode.A_BASELINE:
            raise RuntimeError(
                "Experiment A is the preserved original LAVAD pipeline. Run scripts/01_caption.sh "
                "through scripts/07_eval.sh; the M3VAD runner is for experiments B-E."
            )
        video_id = Path(video.path).name
        captions = _load_json(self.captions_dir / f"{video_id}.json")
        precomputed_summaries = (
            _load_json(self.summaries_dir / f"{video_id}.json")
            if self.summaries_dir
            else None
        )
        fps = _resolve_fps(video_id, self.video_dir, self.config.default_fps)

        memory = self._build_memory(video_id)
        memory.load()
        result_path = self.output_dir / "records" / f"{video_id}.json"
        payload = _load_json(result_path) if result_path.exists() else {
            "video_id": video_id,
            "experiment": self.config.experiment.value,
            "fps": fps,
            "total_frames": video.num_frames,
            "target_frame_interval": self.config.target_frame_interval,
            "offline_centered_windows": True,
            "records": {},
        }
        processed = {int(index) for index in payload.get("records", {})}
        if memory.entries and not processed:
            raise ValueError(
                "Memory state exists without matching inference records; refusing a non-causal resume."
            )
        previous_final_scores = [
            float(payload["records"][str(index)]["final_anomaly_score"])
            for index in sorted(processed)
        ]

        for frame_index in range(0, video.num_frames, self.config.target_frame_interval):
            if frame_index in processed:
                continue
            started = time.perf_counter()
            calls_before = self.scorer.call_count
            summary_calls_before = getattr(self.onevision_summarizer, "call_count", 0)
            frame_caption = _load_caption(captions, frame_index)
            summary, window, cache_hit = self._summary_for_frame(
                video_id=video_id,
                frame_dir=Path(video.path),
                total_frames=video.num_frames,
                frame_index=frame_index,
                fps=fps,
                precomputed=precomputed_summaries,
            )
            frame_paths = [
                Path(video.path) / self.config.imagefile_template.format(index)
                for index in window.sampled_indices
            ]
            target_path = Path(video.path) / self.config.imagefile_template.format(frame_index)
            embeddings = self.encoder.encode(
                target_frame=target_path,
                clip_frames=frame_paths,
                frame_caption=frame_caption,
                summary_10s=summary,
            )

            base_result = self.scorer.score(
                frame_caption=frame_caption,
                summary_10s=summary,
                base_only=True,
            )
            query = QueryEmbeddings(
                visual=embeddings.visual,
                caption=embeddings.caption,
                summary=embeddings.summary,
            )
            retrieved = (
                memory.retrieve(query, query_frame_index=frame_index)
                if self.config.memory.enabled
                else []
            )
            provisional_reliability = (
                embeddings.visual_caption_alignment
                + embeddings.visual_summary_alignment
            ) / 2.0
            support = self.support_calculator.compute(
                retrieved, query_reliability=provisional_reliability
            )
            if self.config.memory.enabled and retrieved:
                final_result = self.scorer.score(
                    frame_caption=frame_caption,
                    summary_10s=summary,
                    retrieved=retrieved,
                    support=support,
                    robust_memory=self.config.memory.mode == "robust",
                    base_only=False,
                )
            else:
                final_result = base_result

            stability = _temporal_stability(
                previous_final_scores, final_result.anomaly_score
            )
            evidence_conflict = bool(
                base_result.evidence_conflict or final_result.evidence_conflict
            )
            overall_reliability = max(
                0.0,
                min(
                    1.0,
                    (
                        embeddings.visual_caption_alignment
                        + embeddings.visual_summary_alignment
                        + stability
                        + (0.0 if evidence_conflict else 1.0)
                    )
                    / 4.0,
                ),
            )
            reliability = {
                "visual_caption_alignment": embeddings.visual_caption_alignment,
                "visual_summary_alignment": embeddings.visual_summary_alignment,
                "temporal_stability": stability,
                "evidence_conflict": evidence_conflict,
                "overall_reliability": overall_reliability,
            }

            update = {
                "candidate": False,
                "accepted": False,
                "status": "not_considered",
                "reasons": ["final_score_above_threshold"],
            }
            if (
                self.config.memory.enabled
                and final_result.anomaly_score <= self.config.memory.score_threshold
            ):
                entry = MemoryEntry.create(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    frame_caption=frame_caption,
                    summary_10s=summary,
                    base_anomaly_score=base_result.anomaly_score,
                    final_anomaly_score=final_result.anomaly_score,
                    visual_embedding=embeddings.visual,
                    caption_embedding=embeddings.caption,
                    summary_embedding=embeddings.summary,
                    reliability_metrics=reliability,
                )
                decision = memory.add_candidate(entry, self._build_update_policy())
                update = {
                    "candidate": True,
                    "accepted": decision.accepted,
                    "status": decision.status.value,
                    "reasons": decision.reasons,
                    "memory_id": entry.memory_id,
                }

            previous_final_scores.append(final_result.anomaly_score)
            memory.save()
            payload["records"][str(frame_index)] = {
                "frame_index": frame_index,
                "timestamp": frame_index / fps,
                "frame_caption": frame_caption,
                "summary_10s": summary,
                "temporal_window": asdict(window),
                "summary_cache_hit": cache_hit,
                "base_anomaly_score": base_result.anomaly_score,
                "final_anomaly_score": final_result.anomaly_score,
                "base_reasoning": base_result.to_dict(),
                "final_reasoning": final_result.to_dict(),
                "normality_support": support.to_dict(),
                "retrieved_memory": [item.to_prompt_dict() for item in retrieved],
                "reliability_metrics": reliability,
                "memory_update": update,
                "memory_counts": memory.counts(),
                "anomaly_llm_calls": self.scorer.call_count - calls_before,
                "onevision_calls": getattr(self.onevision_summarizer, "call_count", 0)
                - summary_calls_before,
                "llm_calls": (
                    self.scorer.call_count
                    - calls_before
                    + getattr(self.onevision_summarizer, "call_count", 0)
                    - summary_calls_before
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            self._save_video_outputs(payload, result_path)
        return payload

    def _summary_for_frame(
        self,
        *,
        video_id: str,
        frame_dir: Path,
        total_frames: int,
        frame_index: int,
        fps: float,
        precomputed: Mapping[str, str] | None,
    ) -> tuple[str, TemporalWindow, bool]:
        if precomputed is not None:
            if str(frame_index) not in precomputed:
                raise KeyError(f"precomputed summary missing for frame {frame_index}")
            window = build_centered_window(
                target_frame=frame_index,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=self.config.onevision.duration_seconds,
                num_samples=self.config.onevision.num_samples,
            )
            return str(precomputed[str(frame_index)]), window, True
        if not self.config.onevision.enabled or self.onevision_summarizer is None:
            raise ValueError("OneVision is enabled but no summarizer or precomputed summaries exist")
        return self.onevision_summarizer.summarize(
            video_id=video_id,
            frame_dir=frame_dir,
            total_frames=total_frames,
            target_frame=frame_index,
            fps=fps,
        )

    def _build_memory(self, video_id: str) -> DynamicMemoryBank:
        config = self.config.memory
        retriever = MemoryRetriever(
            top_k=config.top_k,
            stage1_multiplier=config.stage1_multiplier,
            weights=config.weights,
            min_temporal_gap_frames=config.min_temporal_gap_frames,
        )
        return DynamicMemoryBank(
            video_id=video_id,
            capacity=config.capacity,
            duplicate_threshold=config.duplicate_threshold,
            persistence_dir=self.output_dir / "memory",
            retriever=retriever,
        )

    def _build_update_policy(self):
        config = self.config.memory
        if config.mode == "legacy":
            return LegacyMemoryPolicy(config.score_threshold)
        if config.mode == "robust":
            return RobustMemoryPolicy(
                score_threshold=config.score_threshold,
                alignment_threshold=config.alignment_threshold,
                temporal_stability_threshold=config.temporal_stability_threshold,
                overall_reliability_threshold=config.overall_reliability_threshold,
            )
        raise ValueError("Memory update requested while memory.mode is off")

    def _save_video_outputs(self, payload: dict, result_path: Path) -> None:
        ordered_frames = sorted(int(index) for index in payload["records"])
        final_scores = {
            index: float(payload["records"][str(index)]["final_anomaly_score"])
            for index in ordered_frames
        }
        base_scores = {
            index: float(payload["records"][str(index)]["base_anomaly_score"])
            for index in ordered_frames
        }
        frame_scores = interpolate_target_scores(final_scores, int(payload["total_frames"]))
        _atomic_json_write(result_path, payload)
        _atomic_json_write(
            self.output_dir / "target_scores" / f"{payload['video_id']}.json",
            {
                "video_id": payload["video_id"],
                "base_target_scores": {str(key): value for key, value in base_scores.items()},
                "final_target_scores": {str(key): value for key, value in final_scores.items()},
            },
        )
        _atomic_json_write(
            self.output_dir / "frame_scores" / f"{payload['video_id']}.json",
            {"video_id": payload["video_id"], "frame_scores": frame_scores},
        )


def _build_pipeline(args) -> M3VADPipeline:
    config = M3VADConfig.from_json(args.config)
    if args.mock_models:
        backend = HeuristicMockBackend()
        encoder = HashMultimodalEncoder()
    else:
        if args.llm_backend == "llama2":
            if not args.ckpt_dir or not args.tokenizer_path:
                raise ValueError("Llama 2 requires --ckpt_dir and --tokenizer_path")
            backend = Llama2ChatBackend(
                ckpt_dir=args.ckpt_dir,
                tokenizer_path=args.tokenizer_path,
            )
        else:
            if not args.llm_model:
                raise ValueError("Transformers backend requires --llm_model")
            backend = TransformersChatBackend(model_name=args.llm_model)
        encoder = MultimodalEncoder(
            ImageBindVisualEncoder(),
            SentenceTransformerTextEncoder(args.text_embedding_model),
        )
    scorer = StructuredAnomalyScorer(backend, max_retries=args.max_llm_retries)
    summarizer = None
    if config.onevision.enabled and not args.summaries_dir:
        summarizer = LlavaOneVisionSummarizer(
            cache_dir=Path(args.output_dir) / "onevision_cache",
            model_name=config.onevision.model_name,
            num_samples=config.onevision.num_samples,
            duration_seconds=config.onevision.duration_seconds,
            expected_fps=config.default_fps,
            fps_tolerance=config.fps_tolerance,
            imagefile_template=config.imagefile_template,
            max_new_tokens=config.onevision.max_new_tokens,
            load_in_4bit=config.onevision.load_in_4bit,
        )
    return M3VADPipeline(
        config=config,
        captions_dir=args.captions_dir,
        output_dir=args.output_dir,
        scorer=scorer,
        encoder=encoder,
        onevision_summarizer=summarizer,
        summaries_dir=args.summaries_dir,
        video_dir=args.video_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--annotationfile_path", required=True)
    parser.add_argument("--captions_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_dir")
    parser.add_argument("--summaries_dir")
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--llm_backend", choices=["transformers", "llama2"], default="transformers")
    parser.add_argument("--llm_model")
    parser.add_argument(
        "--text_embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--max_llm_retries", type=int, default=2)
    parser.add_argument(
        "--mock_models",
        action="store_true",
        help="Use deterministic lightweight mocks. Outputs are not valid research metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = _build_pipeline(args)
    videos = [
        VideoRecord(line.strip().split(), args.root_path)
        for line in open(args.annotationfile_path, encoding="utf-8")
        if line.strip()
    ]
    for video in videos:
        pipeline.process_video(video)


if __name__ == "__main__":
    main()
