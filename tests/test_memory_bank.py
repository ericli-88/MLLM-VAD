import json
import tempfile
import unittest
from pathlib import Path

from src.memory.memory_bank import DynamicMemoryBank
from src.memory.memory_entry import MemoryEntry, MemoryStatus
from src.memory.retriever import MemoryRetriever, QueryEmbeddings
from src.memory.similarity import SimilarityWeights
from src.memory.update_policy import LegacyMemoryPolicy, RobustMemoryPolicy


def make_entry(
    frame_index,
    *,
    video_id="video-a",
    base_score=0.1,
    final_score=0.1,
    vector=None,
    reliability=None,
):
    vector = vector or [1.0, 0.0, 0.0]
    reliability = reliability or {
        "visual_caption_alignment": 0.9,
        "visual_summary_alignment": 0.9,
        "temporal_stability": 0.9,
        "evidence_conflict": False,
        "overall_reliability": 0.9,
    }
    return MemoryEntry.create(
        video_id=video_id,
        frame_index=frame_index,
        timestamp=frame_index / 30,
        frame_caption=f"caption {frame_index}",
        summary_10s=f"summary {frame_index}",
        base_anomaly_score=base_score,
        final_anomaly_score=final_score,
        visual_embedding=vector,
        caption_embedding=vector,
        summary_embedding=vector,
        reliability_metrics=reliability,
    )


class MemoryBankTests(unittest.TestCase):
    def test_empty_memory_and_less_than_k_degrade_cleanly(self):
        bank = DynamicMemoryBank(video_id="video-a", retriever=MemoryRetriever(top_k=3))
        query = QueryEmbeddings([1, 0, 0], [1, 0, 0], [1, 0, 0])
        self.assertEqual(bank.retrieve(query, query_frame_index=10), [])
        bank.add_candidate(make_entry(0), LegacyMemoryPolicy())
        result = bank.retrieve(query, query_frame_index=10)
        self.assertEqual(len(result), 1)

    def test_threshold_includes_point_two_but_not_point_three(self):
        bank = DynamicMemoryBank(video_id="video-a")
        decision = bank.add_candidate(
            make_entry(0, base_score=0.2, final_score=0.2), LegacyMemoryPolicy()
        )
        self.assertTrue(decision.accepted)
        rejected = bank.add_candidate(
            make_entry(10, base_score=0.3, final_score=0.3), LegacyMemoryPolicy()
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(bank.counts()["trusted"], 1)
        self.assertEqual(bank.counts()["total"], 1)

    def test_current_or_future_frame_cannot_be_retrieved(self):
        bank = DynamicMemoryBank(
            video_id="video-a",
            retriever=MemoryRetriever(top_k=3, min_temporal_gap_frames=0),
        )
        bank.add_candidate(make_entry(10), LegacyMemoryPolicy())
        query = QueryEmbeddings([1, 0, 0], [1, 0, 0], [1, 0, 0])
        self.assertEqual(bank.retrieve(query, query_frame_index=10), [])
        self.assertEqual(len(bank.retrieve(query, query_frame_index=20)), 1)

    def test_robust_policy_keeps_conflict_pending(self):
        metrics = {
            "visual_caption_alignment": 0.9,
            "visual_summary_alignment": 0.9,
            "temporal_stability": 0.9,
            "evidence_conflict": True,
            "overall_reliability": 0.9,
        }
        bank = DynamicMemoryBank(video_id="video-a")
        decision = bank.add_candidate(
            make_entry(0, reliability=metrics), RobustMemoryPolicy()
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(bank.counts()["pending"], 1)
        self.assertEqual(bank.counts()["trusted"], 0)

    def test_duplicate_does_not_enter_trusted_memory(self):
        bank = DynamicMemoryBank(video_id="video-a", duplicate_threshold=0.98)
        policy = RobustMemoryPolicy()
        self.assertTrue(bank.add_candidate(make_entry(0), policy).accepted)
        decision = bank.add_candidate(make_entry(10), policy)
        self.assertFalse(decision.accepted)
        self.assertIn("duplicate_normal_pattern", decision.reasons)
        self.assertEqual(bank.counts()["trusted"], 1)

    def test_duplicate_execution_is_idempotent(self):
        bank = DynamicMemoryBank(video_id="video-a")
        entry = make_entry(0)
        bank.add_candidate(entry, LegacyMemoryPolicy())
        decision = bank.add_candidate(entry, LegacyMemoryPolicy())
        self.assertIn("entry_already_exists", decision.reasons)
        self.assertEqual(bank.counts()["total"], 1)

    def test_cross_video_contamination_is_rejected(self):
        bank = DynamicMemoryBank(video_id="video-a")
        with self.assertRaises(ValueError):
            bank.add_candidate(make_entry(0, video_id="video-b"), LegacyMemoryPolicy())

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = DynamicMemoryBank(video_id="video-a", persistence_dir=directory)
            bank.add_candidate(make_entry(0), LegacyMemoryPolicy())
            bank.save()
            restored = DynamicMemoryBank(video_id="video-a", persistence_dir=directory)
            restored.load()
            self.assertEqual(restored.counts()["trusted"], 1)
            self.assertEqual(restored.entries[0].frame_caption, "caption 0")

    def test_metadata_embedding_id_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "video-a"
            root.mkdir(parents=True)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "video_id": "video-a",
                        "entry_ids": ["missing"],
                        "entries": [{"memory_id": "missing"}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "embeddings.json").write_text("{}", encoding="utf-8")
            bank = DynamicMemoryBank(video_id="video-a", persistence_dir=directory)
            with self.assertRaises(ValueError):
                bank.load()

    def test_vector_index_count_mismatch_is_detected(self):
        class FakeIndex:
            ntotal = 99

        bank = DynamicMemoryBank(video_id="video-a")
        bank.add_candidate(make_entry(0), LegacyMemoryPolicy())
        bank._faiss_indexes = {"visual": FakeIndex()}
        with self.assertRaises(ValueError):
            bank.validate()

    def test_temporal_diversity_can_return_fewer_than_k(self):
        bank = DynamicMemoryBank(
            video_id="video-a",
            retriever=MemoryRetriever(top_k=3, min_temporal_gap_frames=150),
        )
        for index in (0, 10, 20):
            bank.add_candidate(make_entry(index, vector=[1.0, index / 100, 0.0]), LegacyMemoryPolicy())
        result = bank.retrieve(
            QueryEmbeddings([1, 0, 0], [1, 0, 0], [1, 0, 0]),
            query_frame_index=100,
        )
        self.assertEqual(len(result), 1)

    def test_capacity_bounds_pending_and_trusted_entries(self):
        bank = DynamicMemoryBank(video_id="video-a", capacity=2)
        conflict = {
            "visual_caption_alignment": 0.9,
            "visual_summary_alignment": 0.9,
            "temporal_stability": 0.9,
            "evidence_conflict": True,
            "overall_reliability": 0.5,
        }
        for index in (0, 10, 20):
            bank.add_candidate(
                make_entry(index, vector=[1.0, index / 100, 0.0], reliability=conflict),
                RobustMemoryPolicy(),
            )
        self.assertLessEqual(bank.counts()["total"], 2)


if __name__ == "__main__":
    unittest.main()
