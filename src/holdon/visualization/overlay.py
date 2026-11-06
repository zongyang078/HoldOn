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
