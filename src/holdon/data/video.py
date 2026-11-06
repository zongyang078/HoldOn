"""OpenCV helpers for frame-by-frame video reading and basic video metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata reported by the container."""

    width: int
    height: int
    fps: float
    num_frames: int


def _open(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video {path}")
    return cap


def video_info(path: str | Path) -> VideoInfo:
    """Read width, height, fps and frame count from the video header."""
    cap = _open(path)
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Read one BGR frame by 0-based index."""
    cap = _open(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"Frame {index} not readable in {path}")
        return frame
    finally:
        cap.release()


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, BGR frame)`` pairs sequentially from the start of the video."""
    cap = _open(path)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()
