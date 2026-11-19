"""Overlay hold boxes, keypoints and contact state on video frames."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from holdon.data.wayup import Hold

HANDHOLD_COLOR: tuple[int, int, int] = (0, 255, 0)  # BGR
FOOTHOLD_COLOR: tuple[int, int, int] = (255, 128, 0)


def draw_holds(frame: np.ndarray, holds: Iterable[Hold], label: bool = True) -> np.ndarray:
    """Return a copy of ``frame`` with hold boxes drawn (footholds in a second colour)."""
    out = frame.copy()
    for hold in holds:
        color = FOOTHOLD_COLOR if hold.is_foothold else HANDHOLD_COLOR
        cv2.rectangle(out, (hold.x1, hold.y1), (hold.x2, hold.y2), color, 2)
        if label:
            cv2.putText(
                out, hold.hold_id, (hold.x1, max(hold.y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


GT_COLOR: tuple[int, int, int] = (0, 200, 255)  # BGR, ground-truth contact (orange-yellow)
PRED_COLOR: tuple[int, int, int] = (255, 0, 255)  # BGR, predicted contact (magenta)
KEYPOINT_COLOR: tuple[int, int, int] = (0, 0, 255)  # BGR, red


def draw_keypoints(
    frame: np.ndarray, points: Iterable[tuple[float, float]], color: tuple[int, int, int] = KEYPOINT_COLOR,
    radius: int = 5,
) -> np.ndarray:
    """Return a copy of ``frame`` with filled circles at the given (x, y) points; NaN points are skipped."""
    out = frame.copy()
    for x, y in points:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
    return out


def highlight_holds(
    frame: np.ndarray, holds: Iterable[Hold], hold_ids: Iterable[str], color: tuple[int, int, int],
    thickness: int = 4, pad: int = 0,
) -> np.ndarray:
    """Return a copy of ``frame`` with thick boxes around the holds whose id is in ``hold_ids``."""
    wanted = set(hold_ids)
    out = frame.copy()
    for hold in holds:
        if hold.hold_id in wanted:
            cv2.rectangle(out, (hold.x1 - pad, hold.y1 - pad), (hold.x2 + pad, hold.y2 + pad), color, thickness)
    return out
