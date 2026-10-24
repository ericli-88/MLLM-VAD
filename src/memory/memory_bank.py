"""Video-isolated Dynamic Normality Memory Bank."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .memory_entry import MemoryEntry, MemoryStatus
from .persistence import (
    load_memory_entries,
    save_faiss_indexes,
    save_memory_entries,
    validate_saved_faiss_indexes,
)
from .retriever import MemoryRetriever, QueryEmbeddings, RetrievedMemory
from .similarity import cosine_similarity, l2_normalize
from .update_policy import LegacyMemoryPolicy, RobustMemoryPolicy, UpdateDecision


class DynamicMemoryBank:
    def __init__(
        self,
        *,
        video_id: str,
        capacity: int = 500,
        duplicate_threshold: float = 0.985,
        persistence_dir: str | Path | None = None,
        retriever: MemoryRetriever | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("memory capacity must be positive")
        self.video_id = video_id
        self.capacity = capacity
        self.duplicate_threshold = duplicate_threshold
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None
        self.retriever = retriever or MemoryRetriever()
        self._entries: Dict[str, MemoryEntry] = {}
        self._faiss_indexes: Dict[str, object] = {}
        self._faiss_ids: List[str] = []

    @property
    def entries(self) -> List[MemoryEntry]:
        return sorted(self._entries.values(), key=lambda item: item.frame_index)

    @property
    def trusted_entries(self) -> List[MemoryEntry]:
        return [entry for entry in self.entries if entry.status == MemoryStatus.TRUSTED]

    @property
    def pending_entries(self) -> List[MemoryEntry]:
        return [entry for entry in self.entries if entry.status == MemoryStatus.PENDING]

    def counts(self) -> dict:
        return {
            "total": len(self._entries),
            "pending": len(self.pending_entries),
            "trusted": len(self.trusted_entries),
        }

    def add_candidate(
        self,
        entry: MemoryEntry,
        policy: LegacyMemoryPolicy | RobustMemoryPolicy,
    ) -> UpdateDecision:
        if entry.video_id != self.video_id:
            raise ValueError("cross-video memory insertion is forbidden")
        existing = self._entries.get(entry.memory_id)
        if existing is not None:
            return UpdateDecision(
                existing.status == MemoryStatus.TRUSTED,
                existing.status,
                ["entry_already_exists"],
            )

        entry.status = MemoryStatus.PENDING
        is_duplicate = self._is_duplicate(entry)
        decision = policy.evaluate(entry, is_duplicate=is_duplicate)
        entry.status = decision.status
        if decision.status == MemoryStatus.REJECTED:
            return decision
        self._entries[entry.memory_id] = entry
        self._enforce_capacity()
        if entry.status == MemoryStatus.TRUSTED:
            self._rebuild_indexes()
        return decision

    def remove(self, memory_id: str) -> bool:
        if memory_id not in self._entries:
            return False
        del self._entries[memory_id]
        self._rebuild_indexes()
        return True

    def retrieve(
        self,
        query: QueryEmbeddings,
        *,
        query_frame_index: int,
        exclude_memory_id: str | None = None,
    ) -> List[RetrievedMemory]:
        candidates = self._faiss_stage1_candidates(query)
        return self.retriever.retrieve(
            candidates,
            query,
            video_id=self.video_id,
            query_frame_index=query_frame_index,
            exclude_memory_id=exclude_memory_id,
        )

    def save(self) -> None:
        if self.persistence_dir is None:
            return
        directory = self.persistence_dir / self.video_id
        directory.mkdir(parents=True, exist_ok=True)
        save_memory_entries(directory, self.video_id, self.entries)
        save_faiss_indexes(directory, self._faiss_indexes, self._faiss_ids)

    def load(self) -> None:
        if self.persistence_dir is None:
            return
        directory = self.persistence_dir / self.video_id
        entries = load_memory_entries(directory, self.video_id)
        self._entries = {entry.memory_id: entry for entry in entries}
        self.validate()
        self._rebuild_indexes()
        validate_saved_faiss_indexes(directory, self._faiss_ids)

    def validate(self) -> None:
        ids = [entry.memory_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate Memory Entry IDs detected")
        for entry in self.entries:
            if entry.video_id != self.video_id:
                raise ValueError("cross-video entry detected")
        trusted = self.trusted_entries
        for field in ("visual_embedding", "caption_embedding", "summary_embedding"):
            dimensions = {len(getattr(entry, field)) for entry in trusted}
            if len(dimensions) > 1:
                raise ValueError(f"inconsistent {field} dimensions")
        for index in self._faiss_indexes.values():
            if getattr(index, "ntotal", len(trusted)) != len(trusted):
                raise ValueError("FAISS index and trusted Memory Entries are inconsistent")

    def _is_duplicate(self, entry: MemoryEntry) -> bool:
        for known in self.trusted_entries:
            caption = cosine_similarity(entry.caption_embedding, known.caption_embedding)
            summary = cosine_similarity(entry.summary_embedding, known.summary_embedding)
            if min(caption, summary) >= self.duplicate_threshold:
                return True
        return False

    def _enforce_capacity(self) -> None:
        while len(self._entries) > self.capacity and self.pending_entries:
            remove_entry = min(
                self.pending_entries,
                key=lambda item: (
                    float(item.reliability_metrics.get("overall_reliability", 0.0)),
                    item.frame_index,
                ),
            )
            del self._entries[remove_entry.memory_id]
        trusted = self.trusted_entries
        while len(self._entries) > self.capacity and trusted:
            redundancy_scores = []
            for entry in trusted:
                others = [item for item in trusted if item.memory_id != entry.memory_id]
                redundancy = max(
                    (
                        0.5 * cosine_similarity(entry.caption_embedding, item.caption_embedding)
                        + 0.5 * cosine_similarity(entry.summary_embedding, item.summary_embedding)
                        for item in others
                    ),
                    default=0.0,
                )
                reliability = float(entry.reliability_metrics.get("overall_reliability", 1.0))
                redundancy_scores.append((redundancy - 0.1 * reliability, entry.frame_index, entry))
            _, _, remove_entry = max(redundancy_scores, key=lambda item: (item[0], item[1]))
            del self._entries[remove_entry.memory_id]
            trusted = self.trusted_entries

    def _rebuild_indexes(self) -> None:
        trusted = self.trusted_entries
        self._faiss_ids = [entry.memory_id for entry in trusted]
        self._faiss_indexes = {}
        if not trusted:
            return
        try:
            import faiss
            import numpy as np
        except ImportError:
            return
        for name, field in (
            ("visual", "visual_embedding"),
            ("caption", "caption_embedding"),
            ("summary", "summary_embedding"),
        ):
            matrix = np.asarray(
                [l2_normalize(getattr(entry, field)) for entry in trusted], dtype="float32"
            )
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss_indexes[name] = index

    def _faiss_stage1_candidates(self, query: QueryEmbeddings) -> List[MemoryEntry]:
        trusted = self.trusted_entries
        if self.retriever.weights.visual > 0:
            index_name = "visual"
            query_vector = query.visual
        else:
            # Text-only ablation uses temporal semantics for its FAISS candidate pool.
            index_name = "summary"
            query_vector = query.summary
        index = self._faiss_indexes.get(index_name)
        if index is None or not trusted:
            return trusted
        try:
            import numpy as np
        except ImportError:
            return trusted
        query_matrix = np.asarray([l2_normalize(list(query_vector))], dtype="float32")
        candidate_count = min(
            len(trusted), self.retriever.top_k * self.retriever.stage1_multiplier
        )
        _, indices = index.search(query_matrix, candidate_count)
        by_id = self._entries
        return [by_id[self._faiss_ids[int(index_id)]] for index_id in indices[0] if index_id >= 0]
