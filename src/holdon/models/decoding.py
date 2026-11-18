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
