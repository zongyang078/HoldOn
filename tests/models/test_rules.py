"""Tests for the threshold-plus-duration contact rule and the Track 1 baseline."""

from __future__ import annotations

import numpy as np

from holdon.data.external import Pose2D
from holdon.data.wayup import Hold
from holdon.models.rules import contact_masks, events_from_distances, track1_events

NAMES = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")


def test_contact_masks_nearest_vs_all():
    dist = np.array([[0, 0, 5], [3, 1, 9], [np.nan, 8, 8]], dtype=float)
    tie = np.array([[2, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    nearest = contact_masks(dist, threshold=2, assign="nearest", tiebreak=tie)
    assert nearest.tolist() == [[False, True, False], [False, True, False], [False, False, False]]
    assert contact_masks(dist, threshold=2, assign="all")[0].tolist() == [True, True, False]


def test_events_from_distances_per_hold():
    t = 30
    dist = np.full((t, 2), 50.0)
    dist[0:15, 0] = 0
    dist[20:30, 1] = 0  # only 10 frames: dropped
    events = events_from_distances(dist, ["a", "b"], video_id="v", limb="right_foot",
                                   threshold=0, min_length=13, max_gap=0)
    assert [(e.hold_id, e.start_frame, e.end_frame, e.score) for e in events] == [("a", 0, 14, 1.0)]


def _pose(left_wrist_xy):
    t = len(left_wrist_xy)
    kpts = np.full((t, len(NAMES), 2), 500.0, dtype=np.float32)  # other limbs far from any hold
    kpts[:, 0] = left_wrist_xy
    return Pose2D(
        video_id="p0_green", model="vitpose-test", keypoint_set="coco17", keypoint_names=NAMES,
        keypoints=kpts, scores=np.ones((t, len(NAMES)), np.float32),
        bbox=np.zeros((t, 4), np.float32), num_persons=np.ones(t, np.int32),
        all_boxes=np.zeros((t, 1, 4), np.float32), route_region=np.zeros(4, np.float32),
        fps=25.0, width=720, height=1280,
    )


CFG = {
    "keypoints": "wrist_ankle",
    "min_score": {"vitpose": 0.3},
    "smoothing": {"method": "median", "window": 3, "max_gap": 2},
    "distance_px": 0,
    "assign": "nearest",
    "min_duration_s": 0.5,
    "merge_gap_frames": 0,
}


def test_track1_hold_then_pass_through():
    holds = [Hold("0", 0, 0, 20, 20, False), Hold("1", 100, 0, 120, 20, False)]
    xy = np.full((60, 2), 60.0)
    xy[5:25] = (10, 10)  # 20 frames on hold 0 -> contact
    xy[40:45] = (110, 10)  # 5 frames over hold 1 (hand passing) -> no contact
    events = track1_events(_pose(xy), holds, CFG)
    assert [(e.limb, e.hold_id, e.start_frame, e.end_frame) for e in events] == [("left_hand", "0", 5, 24)]
