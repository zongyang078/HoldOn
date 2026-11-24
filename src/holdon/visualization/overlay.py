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


def contact_panel(
    width: int, gt: dict[str, str | None], pred: dict[str, str | None], header: str = "",
    limbs: Iterable[str] = ("left_hand", "right_hand", "left_foot", "right_foot"),
) -> np.ndarray:
    """White status band: one row per limb with the labeled and the predicted hold (``-`` = none).

    A row is green when both agree (same hold, or both none) and red otherwise.

    Args:
        width: band width in pixels (usually the frame width).
        gt: limb -> labeled hold id at this frame, or None.
        pred: limb -> predicted hold id at this frame, or None.
        header: optional first line, e.g. video id and time.
    """
    font, scale, step = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 26
    limbs = list(limbs)
    rows = ([header] if header else []) + [""] * len(limbs)
    band = np.full((step * len(rows) + 10, width, 3), 255, np.uint8)
    y = step
    if header:
        cv2.putText(band, header, (8, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += step
    for limb in limbs:
        g, p = gt.get(limb), pred.get(limb)
        color = (0, 140, 0) if g == p else (0, 0, 220)
        text = f"{limb.replace('_', ' '):<10}  labeled {g or '-':>3}   Track 1 {p or '-':>3}"
        cv2.putText(band, text, (8, y), font, scale, color, 1, cv2.LINE_AA)
        y += step
    return band
