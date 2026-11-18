"""Threshold-plus-duration contact rules shared by Track 1 (2D overlap) and Track 3 (3D distance).

The generic rule: a limb is in contact with a hold in a frame if its distance to the hold is at most
a threshold (optionally only for the nearest hold), and contact frames are turned into events that
last at least a minimum duration (``holdon.models.decoding``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from holdon.data.external import Pose2D
from holdon.data.wayup import Hold
from holdon.events import LIMBS, ContactEvent
from holdon.features.overlap2d import hold_boxes, min_distance
from holdon.models.decoding import decode_events, min_frames
from holdon.preprocessing.smoothing import smooth_track

_FINGERTIPS = ("thumb4", "forefinger4", "middle_finger4", "ring_finger4", "pinky_finger4")
_FOOT = ("big_toe", "small_toe", "heel")

# Keypoints that represent each limb; the limb touches a hold if any of its points does.
LIMB_KEYPOINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "wrist_ankle": {
        "left_hand": ("left_wrist",),
        "right_hand": ("right_wrist",),
        "left_foot": ("left_ankle",),
        "right_foot": ("right_ankle",),
    },
    "fingertip_toe": {
        "left_hand": tuple(f"left_{f}" for f in _FINGERTIPS),
        "right_hand": tuple(f"right_{f}" for f in _FINGERTIPS),
        "left_foot": tuple(f"left_{f}" for f in _FOOT),
        "right_foot": tuple(f"right_{f}" for f in _FOOT),
    },
}


def contact_masks(
    dist: np.ndarray,
    threshold: float,
    assign: str = "nearest",
    tiebreak: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame contact decision for one limb against every hold.

    Args:
        dist: (T, H) limb-to-hold distances (pixels in 2D, metres in 3D); NaN = unknown.
        threshold: contact if ``dist <= threshold``.
        assign: ``"nearest"`` keeps only the closest qualifying hold per frame (a limb touches one
            hold at a time; avoids duplicate events on overlapping boxes); ``"all"`` keeps every one.
        tiebreak: optional (T, H) secondary key for equal distances (e.g. distance to box centre;
            several boxes give distance 0 when they overlap).

    Returns:
        (T, H) boolean mask.
    """
    dist = np.asarray(dist, dtype=np.float64)
    ok = np.nan_to_num(dist, nan=np.inf) <= threshold
    if assign == "all":
        return ok
    if assign != "nearest":
        raise ValueError(f"Unknown assign mode {assign!r}")
    key = np.where(ok, np.nan_to_num(dist, nan=np.inf), np.inf)
    if tiebreak is not None:
        key = key + 1e-6 * np.where(ok, np.asarray(tiebreak, dtype=np.float64), 0.0)
    best = np.argmin(key, axis=1)
    out = np.zeros_like(ok)
    rows = np.flatnonzero(ok.any(axis=1))
    out[rows, best[rows]] = True
    return out


def events_from_distances(
    dist: np.ndarray,
    hold_ids: Sequence[str],
    *,
    video_id: str,
    limb: str,
    threshold: float,
    min_length: int,
    max_gap: int,
    assign: str = "nearest",
    tiebreak: np.ndarray | None = None,
) -> list[ContactEvent]:
    """Apply the threshold rule to (T, H) distances of one limb and decode events per hold.

    The event score is the mean of ``1 / (1 + distance)`` over its contact frames, so closer and
    steadier contacts rank first in greedy matching.
    """
    masks = contact_masks(dist, threshold, assign, tiebreak)
    closeness = 1.0 / (1.0 + np.nan_to_num(np.asarray(dist, dtype=np.float64), nan=np.inf))
    events: list[ContactEvent] = []
    for h, hold_id in enumerate(hold_ids):
        if masks[:, h].any():
            events += decode_events(
                masks[:, h], video_id=video_id, limb=limb, hold_id=hold_id,
                min_length=min_length, max_gap=max_gap, frame_scores=closeness[:, h],
            )
    return events


def min_score_for(model: str, table: Mapping[str, float]) -> float:
    """Pick the keypoint confidence threshold for a pose model from ``{model prefix: threshold}``."""
    for prefix, value in table.items():
        if model.startswith(prefix):
            return float(value)
    raise KeyError(f"No min_score configured for pose model {model!r} (have {list(table)})")


def track1_events(pose: Pose2D, holds: Sequence[Hold], cfg: Mapping[str, Any]) -> list[ContactEvent]:
    """Track 1 baseline: a limb keypoint inside (or within ``distance_px`` of) a hold box for at
    least ``min_duration_s`` seconds is a contact. ``cfg`` is ``configs/track1.yaml``."""
    points = LIMB_KEYPOINTS[cfg["keypoints"]]
    sm = cfg["smoothing"]
    min_score = min_score_for(pose.model, cfg["min_score"])
    boxes = hold_boxes(holds)
    hold_ids = [h.hold_id for h in holds]
    centres = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    min_length = min_frames(cfg["min_duration_s"], pose.fps)

    events: list[ContactEvent] = []
    for limb in LIMBS:
        tracks = []
        for name in points[limb]:
            xy, score = pose.track(name)
            tracks.append(smooth_track(xy, score, min_score=min_score, max_gap=sm["max_gap"],
                                       method=sm["method"], window=sm["window"]))
        dist = min_distance(tracks, boxes)
        mean_xy = np.nanmean(np.stack(tracks), axis=0) if len(tracks) > 1 else tracks[0]
        tiebreak = np.linalg.norm(mean_xy[:, None, :] - centres[None], axis=-1)
        events += events_from_distances(
            dist, hold_ids, video_id=pose.video_id, limb=limb, threshold=cfg["distance_px"],
            min_length=min_length, max_gap=cfg["merge_gap_frames"], assign=cfg["assign"],
            tiebreak=np.nan_to_num(tiebreak, nan=0.0),
        )
    return sorted(events, key=lambda e: (e.limb, e.start_frame, e.hold_id))
