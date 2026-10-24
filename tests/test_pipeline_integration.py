import json
import tempfile
import unittest
from pathlib import Path

from src.config import M3VADConfig
from src.data.video_record import VideoRecord
from src.memory.multimodal_encoder import EmbeddingBundle
from src.pipeline.m3vad_pipeline import M3VADPipeline
from src.scoring.structured_scorer import StructuredAnomalyScorer


class AlwaysNormalBackend:
    def generate(self, system_prompt, user_prompt):
        has_memory = '"retrieved_normal_examples": []' not in user_prompt
        return json.dumps(
            {
                "anomaly_score": 0.1,
                "visual_evidence": "No dangerous object is visible.",
                "temporal_evidence": "The activity remains stable.",
                "memory_comparison": "A prior example is similar." if has_memory else "Memory is unavailable.",
                "memory_used": has_memory,
                "evidence_conflict": False,
            }
        )


class FixedEncoder:
    def encode(self, **kwargs):
        return EmbeddingBundle(
            visual=[1.0, 0.0, 0.0],
            caption=[1.0, 0.0, 0.0],
            summary=[1.0, 0.0, 0.0],
            visual_caption_alignment=0.9,
            visual_summary_alignment=0.9,
        )


class PipelineIntegrationTests(unittest.TestCase):
    def test_mock_pipeline_writes_sparse_and_frame_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            video_frames = frames / "video-a"
            video_frames.mkdir(parents=True)
            captions_dir = root / "captions"
            summaries_dir = root / "summaries"
            output_dir = root / "output"
            captions_dir.mkdir()
            summaries_dir.mkdir()
            captions = {str(index): "a person walks" for index in (0, 10, 20)}
            summaries = {str(index): "a person continues walking" for index in (0, 10, 20)}
            (captions_dir / "video-a.json").write_text(json.dumps(captions), encoding="utf-8")
            (summaries_dir / "video-a.json").write_text(json.dumps(summaries), encoding="utf-8")

            config = M3VADConfig.from_dict(
                {
                    "experiment": "E",
                    "target_frame_interval": 10,
                    "onevision": {"enabled": True, "num_samples": 4},
                    "memory": {
                        "enabled": True,
                        "mode": "robust",
                        "retrieval": "multimodal",
                        "min_temporal_gap_frames": 0,
                    },
                }
            )
            pipeline = M3VADPipeline(
                config=config,
                captions_dir=captions_dir,
                output_dir=output_dir,
                scorer=StructuredAnomalyScorer(AlwaysNormalBackend(), max_retries=0),
                encoder=FixedEncoder(),
                onevision_summarizer=None,
                summaries_dir=summaries_dir,
            )
            video = VideoRecord(["video-a", "0", "20", "0"], str(frames))
            payload = pipeline.process_video(video)

            self.assertEqual(sorted(payload["records"]), ["0", "10", "20"])
            self.assertEqual(payload["records"]["0"]["memory_counts"]["trusted"], 1)
            self.assertGreaterEqual(len(payload["records"]["10"]["retrieved_memory"]), 1)
            target_output = json.loads(
                (output_dir / "target_scores" / "video-a.json").read_text(encoding="utf-8")
            )
            frame_output = json.loads(
                (output_dir / "frame_scores" / "video-a.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(target_output["final_target_scores"]), 3)
            self.assertEqual(len(frame_output["frame_scores"]), 21)

            resumed = pipeline.process_video(video)
            self.assertEqual(len(resumed["records"]), 3)
            memory_metadata = json.loads(
                (output_dir / "memory" / "video-a" / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                len(memory_metadata["entry_ids"]), len(set(memory_metadata["entry_ids"]))
            )


if __name__ == "__main__":
    unittest.main()
