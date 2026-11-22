"""Tests for the pure-NumPy helpers of external/wham/run_wham.py (no WHAM, no GPU, synthetic data)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "external" / "wham" / "run_wham.py"
_spec = importlib.util.spec_from_file_location("run_wham", _PATH)
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)


def _scores(valid: list[bool], k: int = 17) -> np.ndarray:
    return np.array([[0.9 if v else 0.0] * k for v in valid])


def test_frame_is_valid_needs_six_visible_keypoints():
    s = np.zeros((3, 17))
    s[0, :6] = 0.5
    s[1, :5] = 0.5
    s[2, :6] = np.nan
    assert rw.frame_is_valid(s).tolist() == [True, False, False]


def test_find_segments_bridges_short_gaps_and_drops_short_runs():
    valid = np.zeros(200, bool)
    valid[0:40] = True  # gap 40..44 (5 frames) bridged
    valid[45:80] = True
    valid[100:120] = True  # 20 frames: too short
    valid[150:200] = True
    assert rw.find_segments(valid, max_gap=5) == [(0, 79), (150, 199)]
    assert rw.find_segments(valid, max_gap=4) == [(0, 39), (45, 79), (150, 199)]
    assert rw.find_segments(np.zeros(10, bool), max_gap=5) == []


def test_bbox_from_keypoints_matches_wham_rule():
    kp = np.array([[[100.0, 200.0], [300.0, 250.0], [0.0, 0.0]]])
    s = np.array([[0.9, 0.9, 0.1]])  # third point invisible, ignored
    cx, cy, scale = rw.bbox_from_keypoints(kp, s)[0]
    assert (cx, cy) == (200.0, 225.0)
    assert scale == pytest.approx(200 * 1.2 / 200)
    assert np.isnan(rw.bbox_from_keypoints(kp, np.zeros((1, 3)))).all()


def test_interpolate_nan_linear_inside_nearest_at_edges():
    x = np.array([[np.nan], [0.0], [np.nan], [4.0], [np.nan]])
    assert rw.interpolate_nan(x)[:, 0].tolist() == [0.0, 0.0, 2.0, 4.0, 4.0]


def test_median_filter_removes_spike_keeps_edges():
    x = np.array([1.0, 1.0, 9.0, 1.0, 1.0, 5.0])
    out = rw.median_filter_1d(x, 3)
    assert out.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]


def test_prepare_segment_masks_bridged_frames():
    t = 5
    kp = np.tile(np.linspace(0, 160, 17)[None, :, None], (t, 1, 2)) + np.arange(t)[:, None, None] * 10.0
    valid = [True, True, False, True, True]
    kp[2] = np.nan
    bbox, keypoints = rw.prepare_segment(kp, _scores(valid), fps=25.0)
    assert bbox.shape == (t, 3) and keypoints.shape == (t, 17, 3)
    assert not np.isnan(bbox).any() and not np.isnan(keypoints).any()
    assert (keypoints[2, :, 2] == 0).all() and (keypoints[1, :, 2] == 0.9).all()
    assert keypoints[2, 0, 0] == pytest.approx(20.0)  # interpolated between frames 1 and 3


def test_limb_vertex_ids_select_beyond_wrist_and_ankle():
    # 25 "joints" on a line; vertices skinned to wrist 20 / hand 22, one before and one beyond the wrist.
    v = np.array([[1.0, 0, 0], [3.0, 0, 0], [3.0, 0, 0], [0.0, -1, 0], [0.0, -3, 0]])
    joints = np.zeros((24, 3))
    joints[18] = [0, 0, 0]  # left elbow
    joints[20] = [2, 0, 0]  # left wrist
    joints[4] = [0, 0, 0]  # left knee
    joints[7] = [0, -2, 0]  # left ankle
    # Regressor that reproduces these joints exactly: put each joint into a dedicated "vertex" mix.
    v_all = np.vstack([v, joints])
    j_reg = np.hstack([np.zeros((24, 5)), np.eye(24)])
    w = np.zeros((len(v_all), 24))
    w[0, 20] = w[1, 22] = w[2, 0] = w[3, 7] = w[4, 10] = 1  # vertex 2 is skinned to the pelvis
    w[5:, 0] = 1
    ids = rw.limb_vertex_ids(v_all, j_reg, w)
    assert ids["left_hand"].tolist() == [1]  # vertex 0 lies before the wrist
    assert ids["left_foot"].tolist() == [4]  # vertex 3 lies above the ankle


def test_scatter_and_reprojection():
    pts = np.array([[[0.0, 0.0, 2.0]], [[1.0, 0.0, 2.0]]])
    full = rw.scatter_to_frames(pts, np.array([1, 3]), 4)
    assert np.isnan(full[[0, 2]]).all() and full[3, 0, 0] == 1.0
    kp2d = rw.project(full, 100.0, (50.0, 60.0))
    assert kp2d[3, 0].tolist() == [100.0, 60.0]
    scores = np.ones((4, 1))
    err = rw.reprojection_error(full, kp2d + [3.0, 4.0], scores, 100.0, (50.0, 60.0))
    assert np.isnan(err[0]) and err[1] == pytest.approx(5.0)
