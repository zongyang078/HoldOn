"""Read per-video files exported by external models (2D pose, WHAM).

This module never imports MMPose, rtmlib, transformers or WHAM; it only reads their output files.
The 2D pose file format is specified in docs/pose2d_format.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSE2D_FORMAT_VERSION = 1
KEYPOINT_SETS: dict[str, int] = {"coco17": 17, "coco_wholebody133": 133}


@dataclass(frozen=True)
class Pose2D:
    """Per-frame 2D keypoints of the climber in one video.

    Frame ``t`` of every array is frame ``t`` (0-based) of the processed video. Frames without a
    detected person have NaN keypoints and bbox and zero scores.
    """

    video_id: str
    model: str
    keypoint_set: str
    keypoint_names: tuple[str, ...]
    keypoints: np.ndarray  # (T, K, 2) float32, pixel x, y in the processed video
    scores: np.ndarray  # (T, K) float32, model-specific confidence scale
    bbox: np.ndarray  # (T, 4) float32, climber box x1, y1, x2, y2
    num_persons: np.ndarray  # (T,) int, person boxes detected in the frame
    fps: float
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        """Number of frames covered by the file."""
        return int(self.keypoints.shape[0])

    def index(self, name: str) -> int:
        """Column index of a keypoint name, e.g. ``"left_wrist"``."""
        try:
            return self.keypoint_names.index(name)
        except ValueError:
            raise KeyError(f"{name!r} not in keypoint set {self.keypoint_set}") from None

    def track(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xy (T, 2), score (T,))`` for one keypoint."""
        i = self.index(name)
        return self.keypoints[:, i], self.scores[:, i]


def load_pose2d(path: str | Path) -> Pose2D:
    """Load and validate one ``<video_id>.npz`` written by ``external/pose2d/run_pose2d.py``."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != POSE2D_FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {version}, expected {POSE2D_FORMAT_VERSION}")
        pose = Pose2D(
            video_id=str(data["video_id"]),
            model=str(data["model"]),
            keypoint_set=str(data["keypoint_set"]),
            keypoint_names=tuple(str(n) for n in data["keypoint_names"]),
            keypoints=data["keypoints"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            bbox=data["bbox"].astype(np.float32),
            num_persons=data["num_persons"].astype(np.int32),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
    _validate(pose, path)
    return pose


def _validate(pose: Pose2D, path: str | Path) -> None:
    expected = KEYPOINT_SETS.get(pose.keypoint_set)
    if expected is None:
        raise ValueError(f"{path}: unknown keypoint_set {pose.keypoint_set!r}")
    t = pose.num_frames
    k = len(pose.keypoint_names)
    shapes = {
        "keypoints": (pose.keypoints.shape, (t, expected, 2)),
        "scores": (pose.scores.shape, (t, expected)),
        "bbox": (pose.bbox.shape, (t, 4)),
        "num_persons": (pose.num_persons.shape, (t,)),
    }
    if k != expected:
        raise ValueError(f"{path}: {k} keypoint names for {pose.keypoint_set} ({expected} expected)")
    for name, (got, want) in shapes.items():
        if got != want:
            raise ValueError(f"{path}: {name} has shape {got}, expected {want}")


def pose2d_path(pose_dir: str | Path, video_id: str) -> Path:
    """Path of one video's pose file inside a version folder, e.g. ``.../rtmw-x-384_2026-11-19``."""
    return Path(pose_dir) / f"{video_id}.npz"
