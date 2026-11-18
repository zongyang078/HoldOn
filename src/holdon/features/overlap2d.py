"""2D cues: whether a keypoint lies inside a hold box and its distance to the box.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from holdon.data.wayup import Hold


def hold_boxes(holds: Sequence[Hold]) -> np.ndarray:
    """Stack hold boxes into an (H, 4) array ``[x1, y1, x2, y2]``."""
    return np.array([[h.x1, h.y1, h.x2, h.y2] for h in holds], dtype=np.float64).reshape(-1, 4)


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
