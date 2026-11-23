"""2D cues: whether a keypoint lies inside a hold box, its distance to the box, and nearby holds.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from holdon.data.external import Pose2D
from holdon.data.wayup import Hold
from holdon.preprocessing.smoothing import smooth_track


def hold_boxes(holds: Sequence[Hold]) -> np.ndarray:
    """Stack hold boxes into an (H, 4) array ``[x1, y1, x2, y2]``."""
    return np.array([[h.x1, h.y1, h.x2, h.y2] for h in holds], dtype=np.float64).reshape(-1, 4)


def box_centres(boxes: np.ndarray) -> np.ndarray:
    """(H, 2) centres of (H, 4) boxes."""
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])


def point_box_distance(xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Euclidean pixel distance from each point to each box; 0 inside or on the border.

    Args:
        xy: (T, 2) points, NaN where missing.
        boxes: (H, 4) boxes ``[x1, y1, x2, y2]``.

    Returns:
        (T, H) distances, NaN for missing points.
    """
    xy = np.asarray(xy, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    x, y = xy[:, 0:1], xy[:, 1:2]
    dx = np.maximum.reduce([boxes[None, :, 0] - x, np.zeros((len(xy), len(boxes))), x - boxes[None, :, 2]])
    dy = np.maximum.reduce([boxes[None, :, 1] - y, np.zeros((len(xy), len(boxes))), y - boxes[None, :, 3]])
    return np.hypot(dx, dy)


def min_distance(points: Sequence[np.ndarray], boxes: np.ndarray) -> np.ndarray:
    """Per frame and box, the smallest distance over several points of one limb.

    Used when a limb is represented by several keypoints (e.g. five fingertips): the limb touches a
    hold if any of its points does. NaN only where every point is missing.
    """
    stacked = np.stack([point_box_distance(p, boxes) for p in points])  # (P, T, H)
    with np.errstate(all="ignore"):
        all_nan = np.isnan(stacked).all(axis=0)
        out = np.where(all_nan, np.nan, np.nanmin(np.where(np.isnan(stacked), np.inf, stacked), axis=0))
    return out


def smoothed_tracks(
    pose: Pose2D, names: Sequence[str], min_score: float, smoothing: Mapping[str, object],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cleaned (T, 2) tracks of several keypoints plus their (T,) mean confidence.

    ``smoothing`` has ``method``, ``window`` and ``max_gap`` as in ``configs/track1.yaml``.
    """
    tracks, scores = [], []
    for name in names:
        xy, score = pose.track(name)
        tracks.append(smooth_track(xy, score, min_score=min_score, max_gap=int(smoothing["max_gap"]),
                                   method=str(smoothing["method"]), window=int(smoothing["window"])))
        scores.append(score)
    return tracks, np.mean(np.stack(scores), axis=0).astype(np.float64)


def mean_point(tracks: Sequence[np.ndarray]) -> np.ndarray:
    """(T, 2) NaN-aware mean of several (T, 2) tracks (NaN only where all are missing)."""
    if len(tracks) == 1:
        return np.asarray(tracks[0], dtype=np.float64)
    stacked = np.stack(tracks)
    with np.errstate(all="ignore"):
        valid = ~np.isnan(stacked).any(axis=-1, keepdims=True)
        n = valid.sum(axis=0)
        total = np.where(valid, stacked, 0.0).sum(axis=0)
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def distance_rank(dist: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """(T, H) rank of every hold by distance within each frame (0 = nearest; NaN distances last).

    ``tiebreak`` (T, H), e.g. the distance to the box centre, orders holds with equal distance (several
    overlapping boxes give 0).
    """
    key = np.nan_to_num(np.asarray(dist, dtype=np.float64), nan=np.inf)
    if tiebreak is not None:
        key = np.where(np.isfinite(key), key + 1e-6 * np.nan_to_num(tiebreak, nan=0.0), key)
    order = np.argsort(key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(key.shape[1])[None, :].repeat(len(key), 0), axis=1)
    return rank


def candidate_mask(dist: np.ndarray, rank: np.ndarray, radius: float, max_candidates: int) -> np.ndarray:
    """(T, H) limb–hold pairs kept as classifier candidates: within ``radius`` and among the
    ``max_candidates`` nearest holds of the frame. Missing distances are never candidates."""
    d = np.nan_to_num(np.asarray(dist, dtype=np.float64), nan=np.inf)
    return (d <= radius) & (np.asarray(rank) < max_candidates)


def count_within(dist: np.ndarray, radius: float) -> np.ndarray:
    """(T,) number of holds within ``radius`` of the limb in each frame."""
    return (np.nan_to_num(np.asarray(dist, dtype=np.float64), nan=np.inf) <= radius).sum(axis=1)
