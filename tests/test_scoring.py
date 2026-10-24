import json
import unittest

from src.scoring.score_utils import interpolate_target_scores, validate_anomaly_score
from src.scoring.structured_scorer import StructuredAnomalyScorer


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, system_prompt, user_prompt):
        return self.responses.pop(0)


def valid_response(score=0.2):
    return json.dumps(
        {
            "anomaly_score": score,
            "visual_evidence": "A person is walking.",
            "temporal_evidence": "The action is stable.",
            "memory_comparison": "Memory is unavailable.",
            "memory_used": False,
            "evidence_conflict": False,
        }
    )


class StructuredScoringTests(unittest.TestCase):
    def test_invalid_json_is_retried(self):
        scorer = StructuredAnomalyScorer(
            SequenceBackend(["not json", valid_response(0.2)]), max_retries=1
        )
        result = scorer.score(frame_caption="walking", summary_10s="walking", base_only=True)
        self.assertEqual(result.anomaly_score, 0.2)
        self.assertEqual(scorer.call_count, 2)

    def test_illegal_score_is_rejected(self):
        scorer = StructuredAnomalyScorer(
            SequenceBackend([valid_response(0.25)]), max_retries=0
        )
        with self.assertRaises(ValueError):
            scorer.score(frame_caption="walking", summary_10s="walking", base_only=True)

    def test_allowed_score_validation(self):
        self.assertEqual(validate_anomaly_score(0.2), 0.2)
        with self.assertRaises(ValueError):
            validate_anomaly_score(0.21)

    def test_interpolation_preserves_target_values(self):
        scores = interpolate_target_scores({0: 0.0, 10: 1.0, 20: 0.5}, 25)
        self.assertEqual(len(scores), 25)
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[10], 1.0)
        self.assertEqual(scores[20], 0.5)
        self.assertEqual(scores[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
