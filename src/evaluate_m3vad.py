"""Frame-level ROC-AUC/PR-AUC evaluation for saved M3VAD frame scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.video_record import VideoRecord
from src.eval import get_video_labels, temporal_testing_annotations


def evaluate(
    *,
    root_path: str,
    annotationfile_path: str,
    temporal_annotation_file: str,
    frame_scores_dir: str,
    output_path: str,
    normal_label: int = 0,
) -> dict:
    annotations = temporal_testing_annotations(temporal_annotation_file)
    videos = [
        VideoRecord(line.strip().split(), root_path)
        for line in open(annotationfile_path, encoding="utf-8")
        if line.strip()
    ]
    flat_scores = []
    flat_labels = []
    for video in videos:
        video_id = Path(video.path).name
        with open(Path(frame_scores_dir) / f"{video_id}.json", encoding="utf-8") as handle:
            scores = json.load(handle)["frame_scores"]
        if len(scores) != video.num_frames:
            raise ValueError(
                f"{video_id} has {len(scores)} frame scores; expected {video.num_frames}"
            )
        labels = get_video_labels(video, annotations, normal_label)
        flat_scores.extend(float(value) for value in scores)
        flat_labels.extend(int(value != normal_label) for value in labels)

    labels_array = np.asarray(flat_labels, dtype=np.int64)
    scores_array = np.asarray(flat_scores, dtype=np.float64)
    metrics = {
        "roc_auc": float(roc_auc_score(labels_array, scores_array)),
        "pr_auc": float(average_precision_score(labels_array, scores_array)),
        "num_frames": int(len(scores_array)),
        "num_videos": int(len(videos)),
        "score_source": "linearly_interpolated_frame_scores",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--annotationfile_path", required=True)
    parser.add_argument("--temporal_annotation_file", required=True)
    parser.add_argument("--frame_scores_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--normal_label", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(**vars(args))
