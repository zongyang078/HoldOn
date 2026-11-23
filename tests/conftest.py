"""Shared pytest fixtures: tiny synthetic data only, CPU only, no real dataset."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.data.external import Pose2D
from holdon.data.wayup import Hold, LabeledEvent
from holdon.events import LIMBS, ContactEvent
from holdon.models.rules import LIMB_KEYPOINTS


@pytest.fixture
def sample_events() -> list[ContactEvent]:
    """A handful of hand-written events for one synthetic video."""
    return [
        ContactEvent("p0_green", "left_hand", "h1", 10, 30, 0.9),
        ContactEvent("p0_green", "right_foot", "h7", 0, 50, 0.6),
    ]


def make_climb(video_id: str, num_frames: int = 60, seed: int = 0):
    """Synthetic video: two holds; the left hand rests on hold 0 in frames 10-39 and moves elsewhere.

    Returns ``(pose, holds, labeled)``. Every limb keypoint used by Track 1 / Track 2 is present; the
    other limbs hang far from both holds.
    """
    rng = np.random.default_rng(seed)
    names = sorted({n for s in LIMB_KEYPOINTS.values() for limb in LIMBS for n in s[limb]})
    k = len(names)
    kp = np.full((num_frames, k, 2), 500.0, dtype=np.float32)
    path = np.column_stack([np.linspace(0, 300, num_frames), np.full(num_frames, 100.0)])
    path[10:40] = [15.0, 15.0]  # still, inside hold 0 (box 0..30)
    hand = [i for i, n in enumerate(names) if n.startswith("left_") and (n.endswith("4") or n == "left_wrist")]
    kp[:, hand] = (path + rng.normal(0, 0.5, path.shape))[:, None, :]
    pose = Pose2D(
        video_id=video_id, model="rtmw-x-384", keypoint_set="coco_wholebody133", keypoint_names=tuple(names),
        keypoints=kp, scores=np.full((num_frames, k), 5.0, np.float32),
        bbox=np.tile(np.array([0, 0, 200, 400], np.float32), (num_frames, 1)),
        num_persons=np.ones(num_frames, np.int32), all_boxes=np.zeros((num_frames, 1, 4), np.float32),
        route_region=np.full(4, np.nan, np.float32), fps=25.0, width=720, height=1280,
    )
    holds = [Hold("0", 0, 0, 30, 30, False), Hold("1", 200, 80, 230, 120, True)]
    labeled = [LabeledEvent(ContactEvent(video_id, "left_hand", "0", 10, 39), ((10, 19),))]
    return pose, holds, labeled


@pytest.fixture
def track2_cfg() -> dict:
    """Small Track 2 config (configs/track2.yaml structure) that trains in well under a second."""
    return {
        "features": {"overlap2d": True, "stillness": True, "context": True, "distance3d": False},
        "keypoints": {"tips": "fingertip_toe", "joint": "wrist_ankle"},
        "min_score": {"rtmw": 1.5},
        "smoothing": {"method": "median", "window": 3, "max_gap": 2},
        "candidates": {"radius_px": 80, "max_per_limb": 3},
        "context_frames": [2],
        "svm": {"nystroem_components": 20, "gamma": None, "C": 1.0, "max_iter": 2000, "max_train_rows": 500},
        "inner_folds": 2,
        "seed": 0,
        "decoding": {"threshold": 0.0, "min_duration_s": 0.4, "merge_gap_frames": 2},
        "tuning": {"objective": "mean_f1", "grid": {"threshold": [-0.5, 0.0], "min_duration_s": [0.4]}},
    }
