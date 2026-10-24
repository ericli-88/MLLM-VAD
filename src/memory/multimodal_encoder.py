"""Pluggable encoders for visual clips, frame captions, and temporal summaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence

from .similarity import cosine_similarity, cosine_to_support, l2_normalize


@dataclass(frozen=True)
class EmbeddingBundle:
    visual: List[float]
    caption: List[float]
    summary: List[float]
    visual_caption_alignment: float = 0.0
    visual_summary_alignment: float = 0.0


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> List[List[float]]: ...


class VisualEncoder(Protocol):
    def encode(self, target_frame: Path, clip_frames: Sequence[Path]) -> List[float]: ...

    def aligned_text(self, texts: Sequence[str]) -> List[List[float]]: ...


class SentenceTransformerTextEncoder:
    """Sentence Transformer wrapper loaded only when production inference starts."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for M3VAD text embeddings; "
                "install requirements-m3vad.txt"
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(value) for value in vector] for vector in vectors]


class ImageBindVisualEncoder:
    """Reuse the bundled ImageBind model for aligned frame/clip representations."""

    def __init__(self, device: str | None = None, target_weight: float = 0.4) -> None:
        import torch

        from libs.ImageBind.imagebind import data
        from libs.ImageBind.imagebind.models.imagebind_model import ModalityType
        from src.utils.torch_utils import initialize_vlm_model_and_device

        self.torch = torch
        self.data = data
        self.modality = ModalityType
        self.model, detected_device = initialize_vlm_model_and_device()
        self.device = device or detected_device
        self.target_weight = max(0.0, min(1.0, target_weight))

    def encode(self, target_frame: Path, clip_frames: Sequence[Path]) -> List[float]:
        target_inputs = {
            self.modality.VISION: self.data.load_and_transform_vision_data(
                [target_frame], self.device
            )
        }
        clip_inputs = {
            self.modality.VISION: self.data.load_and_transform_video_data(
                [list(clip_frames)], self.device
            )
        }
        with self.torch.no_grad():
            target = self.model(target_inputs)[self.modality.VISION][0].cpu().tolist()
            clip = self.model(clip_inputs)[self.modality.VISION][0].cpu().tolist()
        fused = [
            self.target_weight * target_value + (1.0 - self.target_weight) * clip_value
            for target_value, clip_value in zip(target, clip)
        ]
        return l2_normalize(fused)

    def aligned_text(self, texts: Sequence[str]) -> List[List[float]]:
        inputs = {
            self.modality.TEXT: self.data.load_and_transform_text(list(texts), self.device)
        }
        with self.torch.no_grad():
            vectors = self.model(inputs)[self.modality.TEXT].cpu().tolist()
        return [l2_normalize(vector) for vector in vectors]


class DeterministicHashEncoder:
    """Dependency-free encoder for tests and pipeline smoke runs only."""

    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    def _encode_value(self, value: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = value.lower().split() or [value.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (0.5 + digest[5] / 510.0)
        return l2_normalize(vector)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._encode_value(text) for text in texts]

    def encode_visual(self, target_frame: Path, clip_frames: Sequence[Path]) -> List[float]:
        names = " ".join([target_frame.name, *(frame.name for frame in clip_frames)])
        return self._encode_value(names)

    def aligned_text(self, texts: Sequence[str]) -> List[List[float]]:
        return self.encode(texts)


class MultimodalEncoder:
    def __init__(self, visual_encoder: VisualEncoder, text_encoder: TextEncoder) -> None:
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

    def encode(
        self,
        *,
        target_frame: Path,
        clip_frames: Sequence[Path],
        frame_caption: str,
        summary_10s: str,
    ) -> EmbeddingBundle:
        visual = self.visual_encoder.encode(target_frame, clip_frames)
        caption, summary = self.text_encoder.encode([frame_caption, summary_10s])
        aligned_caption, aligned_summary = self.visual_encoder.aligned_text(
            [frame_caption, summary_10s]
        )
        return EmbeddingBundle(
            visual=l2_normalize(visual),
            caption=l2_normalize(caption),
            summary=l2_normalize(summary),
            visual_caption_alignment=cosine_to_support(
                cosine_similarity(visual, aligned_caption)
            ),
            visual_summary_alignment=cosine_to_support(
                cosine_similarity(visual, aligned_summary)
            ),
        )


class HashMultimodalEncoder(MultimodalEncoder):
    """Convenient mock encoder implementing the production interface."""

    def __init__(self, dimension: int = 32) -> None:
        encoder = DeterministicHashEncoder(dimension)

        class _VisualAdapter:
            def encode(self, target_frame: Path, clip_frames: Sequence[Path]) -> List[float]:
                return encoder.encode_visual(target_frame, clip_frames)

            def aligned_text(self, texts: Sequence[str]) -> List[List[float]]:
                return encoder.aligned_text(texts)

        super().__init__(_VisualAdapter(), encoder)
