"""Atomic persistence for memory metadata, embeddings, and optional FAISS indexes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping

from .memory_entry import MemoryEntry


SCHEMA_VERSION = 1


def _atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_memory_entries(directory: Path, video_id: str, entries: Iterable[MemoryEntry]) -> None:
    entries = list(entries)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "entry_ids": [entry.memory_id for entry in entries],
        "entries": [entry.to_dict(include_embeddings=False) for entry in entries],
    }
    embeddings = {
        entry.memory_id: {
            "visual_embedding": entry.visual_embedding,
            "caption_embedding": entry.caption_embedding,
            "summary_embedding": entry.summary_embedding,
        }
        for entry in entries
    }
    _atomic_json_write(directory / "metadata.json", metadata)
    _atomic_json_write(directory / "embeddings.json", embeddings)


def load_memory_entries(directory: Path, expected_video_id: str) -> List[MemoryEntry]:
    metadata_path = directory / "metadata.json"
    embeddings_path = directory / "embeddings.json"
    if not metadata_path.exists() and not embeddings_path.exists():
        return []
    if not metadata_path.exists() or not embeddings_path.exists():
        raise ValueError("memory persistence is incomplete: metadata/embeddings mismatch")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    with open(embeddings_path, encoding="utf-8") as handle:
        embeddings = json.load(handle)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported memory persistence schema")
    if metadata.get("video_id") != expected_video_id:
        raise ValueError("memory video_id does not match the requested video")
    entry_ids = metadata.get("entry_ids", [])
    if entry_ids != [item.get("memory_id") for item in metadata.get("entries", [])]:
        raise ValueError("memory entry order and metadata IDs are inconsistent")
    if set(entry_ids) != set(embeddings):
        raise ValueError("memory index and entry IDs are inconsistent")

    result: List[MemoryEntry] = []
    for item in metadata.get("entries", []):
        memory_id = item["memory_id"]
        merged = dict(item)
        merged.update(embeddings[memory_id])
        result.append(MemoryEntry.from_dict(merged))
    return result


def save_faiss_indexes(
    directory: Path, indexes: Mapping[str, object], index_ids: List[str]
) -> None:
    try:
        import faiss
    except ImportError:
        return
    for stale_path in directory.glob("*.faiss"):
        stale_path.unlink()
    for name, index in indexes.items():
        faiss.write_index(index, str(directory / f"{name}.faiss"))
    _atomic_json_write(directory / "faiss_ids.json", index_ids)


def validate_saved_faiss_indexes(directory: Path, expected_ids: List[str]) -> None:
    paths = list(directory.glob("*.faiss"))
    if not paths:
        return
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required to validate saved memory indexes") from exc
    ids_path = directory / "faiss_ids.json"
    if not ids_path.exists():
        raise ValueError("FAISS indexes exist without their Memory Entry ID mapping")
    with open(ids_path, encoding="utf-8") as handle:
        saved_ids = json.load(handle)
    if saved_ids != expected_ids:
        raise ValueError("FAISS index ID mapping and trusted Memory Entries are inconsistent")
    for path in paths:
        index = faiss.read_index(str(path))
        if index.ntotal != len(expected_ids):
            raise ValueError(
                f"FAISS index {path.name} has {index.ntotal} vectors; expected {len(expected_ids)}"
            )
