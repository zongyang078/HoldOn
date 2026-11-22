"""Tests for focal length and wall pose from bolt holes, on a synthetic camera."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from holdon.preprocessing import homography as hg

W, H = 720, 1280
WORLD_CM = np.array([[0.0, 0, 0], [135, 0, 0], [0, 510, 0], [135, 510, 0]])


def _camera(focal: float, tilt_deg: float, dist: float = 2.6):
    """Camera 0.85 m above the wall origin, ``dist`` m from the wall, pitched up by ``tilt_deg``."""
    # Wall frame: x right, y up, z towards the camera. Camera frame: x right, y down, z forward.
    base = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])  # looking straight at the wall
    tilt = cv2.Rodrigues(np.array([np.radians(tilt_deg), 0.0, 0.0]))[0]
    R = tilt @ base
    cam_in_wall = np.array([0.6, 0.85, dist])
    t = -R @ cam_in_wall
    return R, t


def _project(R, t, focal, world_cm):
    K = hg.camera_matrix(focal, W, H)
    pts = (R @ hg.wall_points_m(world_cm).T).T + t
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def test_fix_bolt_hole_order_swaps_top_pair_only():
    img = np.array([[150.0, 1000], [500, 1000], [460, 70], [250, 40]])  # top two swapped
    fixed, swapped = hg.fix_bolt_hole_order(img, WORLD_CM)
    assert swapped
    assert fixed[2].tolist() == [250, 40] and fixed[3].tolist() == [460, 70]
    again, swapped2 = hg.fix_bolt_hole_order(fixed, WORLD_CM)
    assert not swapped2 and np.array_equal(again, fixed)


def test_fit_wall_pose_recovers_synthetic_camera():
    R, t = _camera(600, 30)
    img = _project(R, t, 600, WORLD_CM)
    pose, swapped = hg.recording_wall_pose(img[[0, 1, 3, 2]][[0, 1, 3, 2]], WORLD_CM, 600, W, H)
    assert not swapped
    assert pose.rms_px < 1e-3
    np.testing.assert_allclose(pose.R, R, atol=1e-6)
    np.testing.assert_allclose(pose.t, t, atol=1e-6)
    np.testing.assert_allclose(pose.cam_to_wall(np.zeros(3)), [0.6, 0.85, 2.6], atol=1e-6)
    assert pose.camera_side() == 1.0


def test_swapped_points_are_repaired_before_fitting():
    R, t = _camera(600, 30)
    img = _project(R, t, 600, WORLD_CM)
    pose, swapped = hg.recording_wall_pose(img[[0, 1, 3, 2]], WORLD_CM, 600, W, H)
    assert swapped and pose.rms_px < 1e-3


def test_estimate_focal_finds_the_true_value():
    walls = []
    for tilt, dist in [(25, 2.4), (32, 2.7), (38, 2.5)]:
        R, t = _camera(600, tilt, dist)
        walls.append((_project(R, t, 600, WORLD_CM), WORLD_CM))
    focal, cost = hg.estimate_focal(walls, W, H, candidates=np.arange(400.0, 1600.0, 10.0))
    assert focal == pytest.approx(600, abs=10)
    assert cost.argmin() not in (0, len(cost) - 1)


def test_image_to_wall_homography_maps_bolt_holes_back():
    R, t = _camera(600, 30)
    img = _project(R, t, 600, WORLD_CM)
    pose = hg.fit_wall_pose(img, WORLD_CM, 600, W, H)
    wall = hg.apply_homography(hg.image_to_wall_homography(pose), img)
    np.testing.assert_allclose(wall, WORLD_CM[:, :2] / 100, atol=1e-6)
