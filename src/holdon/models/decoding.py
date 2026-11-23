"""Turn per-frame contact scores into ContactEvent intervals (threshold, merge, minimum duration).
"""

from __future__ import annotations

import math

import numpy as np

from holdon.events import ContactEvent


def mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` frame intervals of the True runs in a boolean (T,) mask."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(intervals: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge consecutive intervals separated by at most ``max_gap`` frames (input sorted by start)."""
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def min_frames(duration_s: float, fps: float) -> int:
    """Shortest event length in frames for a minimum duration in seconds (rounded up)."""
    return max(1, math.ceil(duration_s * fps - 1e-9))


def decode_events(
    mask: np.ndarray,
    *,
    video_id: str,
    limb: str,
    hold_id: str,
    min_length: int,
    max_gap: int = 0,
    frame_scores: np.ndarray | None = None,
) -> list[ContactEvent]:
    """Convert one limb–hold contact mask into events.

    Steps: True runs -> merge runs separated by <= ``max_gap`` frames -> drop events shorter than
    ``min_length`` frames. The event score is the mean of ``frame_scores`` over the event's
    contact frames (1.0 if not given); it orders predictions in greedy matching.
    """
    mask = np.asarray(mask, dtype=bool)
    events = []
    for start, end in merge_intervals(mask_to_intervals(mask), max_gap):
        if end - start + 1 < min_length:
            continue
        if frame_scores is None:
            score = 1.0
        else:
            seg = np.asarray(frame_scores, dtype=np.float64)[start : end + 1][mask[start : end + 1]]
            score = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
        events.append(ContactEvent(video_id, limb, hold_id, int(start), int(end), score))
    return events


def best_hold_per_frame(
    frames: np.ndarray, holds: np.ndarray, scores: np.ndarray, num_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """For one limb, the highest-scoring candidate hold in every frame.

    Args:
        frames, holds, scores: (N,) candidate rows of one limb of one video (frame index, hold id,
            classifier score); a frame may have several candidate holds or none.
        num_frames: number of frames of the video.

    Returns:
        ``(best (T,) int, best_score (T,) float, hold_ids)``: index into ``hold_ids`` of the best
        hold (-1 and score -inf where the frame has no candidate).
    """
    hold_ids = sorted(set(np.asarray(holds).tolist()))
    index = {h: i for i, h in enumerate(hold_ids)}
    table = np.full((num_frames, max(len(hold_ids), 1)), -np.inf)
    if len(frames):
        cols = np.array([index[h] for h in holds])
        table[np.asarray(frames, dtype=int), cols] = np.asarray(scores, dtype=np.float64)
    best = np.argmax(table, axis=1)
    best_score = table[np.arange(num_frames), best]
    best = np.where(np.isfinite(best_score), best, -1)
    return best, best_score, hold_ids


def decode_best_holds(
    best: np.ndarray,
    best_score: np.ndarray,
    hold_ids: list[str],
    *,
    video_id: str,
    limb: str,
    threshold: float,
    min_length: int,
    max_gap: int,
) -> list[ContactEvent]:
    """Events from ``best_hold_per_frame``: a frame is a contact with its best hold if the score is
    above ``threshold``; then ``decode_events`` per hold. Event score = mean frame score."""
    on = (best >= 0) & (best_score > threshold)
    events: list[ContactEvent] = []
    for h, hold_id in enumerate(hold_ids):
        mask = on & (best == h)
        if mask.any():
            events += decode_events(mask, video_id=video_id, limb=limb, hold_id=hold_id,
                                    min_length=min_length, max_gap=max_gap, frame_scores=best_score)
    return events
