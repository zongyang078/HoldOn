"""Limb stillness from keypoint velocity (optical flow is planned for Phase 3).

Speeds are in pixels per frame divided by the climber's box height, so a climber high on the wall
(smaller in the image) and one near the ground are on the same scale.
"""

from __future__ import annotations

import numpy as np

from holdon.preprocessing.smoothing import median_filter


def keypoint_speed(xy: np.ndarray) -> np.ndarray:
    """(T,) speed of a (T, 2) track in pixels per frame (central differences, one-sided at the ends).

    NaN where the point or a needed neighbour is missing.
    """
    xy = np.asarray(xy, dtype=np.float64)
    t = len(xy)
    if t < 2:
        return np.full(t, np.nan)
    diff = np.full((t, 2), np.nan)
    diff[1:-1] = (xy[2:] - xy[:-2]) / 2
    diff[0] = xy[1] - xy[0]
    diff[-1] = xy[-1] - xy[-2]
    return np.hypot(diff[:, 0], diff[:, 1])


def body_scale(bbox: np.ndarray, window: int = 25) -> np.ndarray:
    """(T,) climber box height in pixels, median-filtered over ``window`` frames; NaN if no box."""
    bbox = np.asarray(bbox, dtype=np.float64)
    height = (bbox[:, 3] - bbox[:, 1])[:, None]
    height[height <= 0] = np.nan
    return median_filter(height, window)[:, 0]


def normalized_speed(xy: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(T,) keypoint speed divided by the body scale (box heights per frame)."""
    with np.errstate(all="ignore"):
        return keypoint_speed(xy) / np.asarray(scale, dtype=np.float64)
