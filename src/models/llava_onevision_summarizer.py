"""Centered 10-second video summaries with LLaVA-OneVision and resumable caching."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence


DEFAULT_PROMPT = (
    "Describe the events in this video window in chronological order. Focus on people, "
    "objects, actions, interactions, sudden changes, dangerous behavior, and evidence "
    "relevant to anomaly detection. Distinguish the central moment from surrounding context. "
    "Do not assign an anomaly score."
)


def uniform_indices(start: int, end: int, count: int) -> List[int]:
    """Uniform inclusive sampling that always covers both window boundaries."""

    if start < 0 or end < start:
        raise ValueError("invalid inclusive frame range")
    if count <= 0:
        raise ValueError("sample count must be positive")
    length = end - start + 1
    if count >= length:
        return list(range(start, end + 1))
    if count == 1:
        return [start + (length - 1) // 2]
    indices = [round(start + index * (length - 1) / (count - 1)) for index in range(count)]
    result: List[int] = []
    for index in indices:
        if not result or result[-1] != index:
            result.append(index)
    return result


@dataclass(frozen=True)
class TemporalWindow:
    target_frame: int
    start_frame: int
    end_frame: int
    sampled_indices: List[int]
    fps: float
    duration_seconds: float


def build_centered_window(
    *,
    target_frame: int,
    total_frames: int,
    fps: float = 30.0,
    duration_seconds: float = 10.0,
    num_samples: int = 16,
) -> TemporalWindow:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if target_frame < 0 or target_frame >= total_frames:
        raise ValueError("target_frame is outside the video")
    if fps <= 0 or duration_seconds <= 0:
        raise ValueError("fps and duration_seconds must be positive")
    radius = int(round(fps * duration_seconds / 2.0))
    start = max(0, target_frame - radius)
    end = min(total_frames - 1, target_frame + radius)
    return TemporalWindow(
        target_frame=target_frame,
        start_frame=start,
        end_frame=end,
        sampled_indices=uniform_indices(start, end, num_samples),
        fps=float(fps),
        duration_seconds=float(duration_seconds),
    )


class SummaryCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}.json"

    def get(self, video_id: str, key: str) -> dict | None:
        path = self._path(video_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get(key)

    def put(self, video_id: str, key: str, value: dict) -> None:
        path = self._path(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        payload[key] = value
        temporary = path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


class LlavaOneVisionSummarizer:
    def __init__(
        self,
        *,
        cache_dir: str | Path,
        model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_samples: int = 16,
        duration_seconds: float = 10.0,
        expected_fps: float = 30.0,
        fps_tolerance: float = 0.5,
        imagefile_template: str = "{:06d}.jpg",
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ) -> None:
        self.cache = SummaryCache(cache_dir)
        self.model_name = model_name
        self.num_samples = num_samples
        self.duration_seconds = duration_seconds
        self.expected_fps = expected_fps
        self.fps_tolerance = fps_tolerance
        self.imagefile_template = imagefile_template
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self._processor = None
        self._model = None
        self.call_count = 0

    def summarize(
        self,
        *,
        video_id: str,
        frame_dir: str | Path,
        total_frames: int,
        target_frame: int,
        fps: float,
    ) -> tuple[str, TemporalWindow, bool]:
        if abs(float(fps) - self.expected_fps) > self.fps_tolerance:
            warnings.warn(
                f"{video_id} reports {fps:.3f} FPS rather than {self.expected_fps:.3f}; "
                "the 10-second window is converted using the reported FPS.",
                RuntimeWarning,
            )
        window = build_centered_window(
            target_frame=target_frame,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=self.duration_seconds,
            num_samples=self.num_samples,
        )
        cache_key = str(target_frame)
        expected_metadata = {
            "model_name": self.model_name,
            "window": asdict(window),
            "prompt": self.prompt,
        }
        cached = self.cache.get(video_id, cache_key)
        if cached and cached.get("metadata") == expected_metadata:
            return str(cached["summary"]), window, True

        frame_dir = Path(frame_dir)
        frame_paths = [
            frame_dir / self.imagefile_template.format(index) for index in window.sampled_indices
        ]
        missing = [str(path) for path in frame_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing sampled video frames: {missing[:3]}")
        self.call_count += 1
        summary = self._generate(frame_paths)
        self.cache.put(
            video_id,
            cache_key,
            {"summary": summary, "metadata": expected_metadata},
        )
        return summary, window, False

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA-OneVision requires the optional M3VAD dependencies. "
                "Install requirements-m3vad.txt."
            ) from exc

        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {"device_map": self.device_map, "torch_dtype": model_dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"
        self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        ).eval()

    def _generate(self, frame_paths: Sequence[Path]) -> str:
        self._ensure_model()
        import numpy as np
        import torch
        from PIL import Image

        frames = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in frame_paths])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=prompt,
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )
        device = next(self._model.parameters()).device
        inputs = {
            key: value.to(device, dtype=torch.float16)
            if getattr(value, "is_floating_point", lambda: False)()
            else value.to(device)
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self._processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
