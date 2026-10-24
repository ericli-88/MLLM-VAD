"""Discrete anomaly-score validation and frame-level interpolation."""

from __future__ import annotations

from typing import List, Mapping


ALLOWED_SCORES = tuple(round(index / 10, 1) for index in range(11))


def validate_anomaly_score(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid anomaly score")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid anomaly score: {value!r}") from exc
    rounded = round(score, 1)
    if abs(score - rounded) > 1e-8 or rounded not in ALLOWED_SCORES:
        raise ValueError(f"anomaly score must be one of {ALLOWED_SCORES}; got {value!r}")
    return rounded


def interpolate_target_scores(
    target_scores: Mapping[int, float], total_frames: int
) -> List[float]:
    """Linearly interpolate discrete target scores to all frames.

    Target-frame values remain exactly discrete; intermediate frame-level values
    are continuous evaluation values and are saved separately.
    """

    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if total_frames == 0:
        return []
    if not target_scores:
        raise ValueError("at least one target score is required")
    points = sorted((int(index), validate_anomaly_score(score)) for index, score in target_scores.items())
    if points[0][0] < 0 or points[-1][0] >= total_frames:
        raise ValueError("target frame index is outside the video")

    result = [points[0][1]] * total_frames
    for point_index, (left_frame, left_score) in enumerate(points):
        right_frame, right_score = (
            points[point_index + 1] if point_index + 1 < len(points) else (total_frames - 1, left_score)
        )
        span = right_frame - left_frame
        for frame_index in range(left_frame, min(right_frame, total_frames - 1) + 1):
            alpha = 0.0 if span <= 0 else (frame_index - left_frame) / span
            result[frame_index] = float(left_score + alpha * (right_score - left_score))
    last_frame, last_score = points[-1]
    for frame_index in range(last_frame, total_frames):
        result[frame_index] = float(last_score)
    return result
