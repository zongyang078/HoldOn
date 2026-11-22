"""Tests for 3D limb-to-hold distances on a synthetic camera, wall and body."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.data.external import load_distance3d
from holdon.data.wayup import Hold
from holdon.events import LIMBS
from holdon.features import distance3d as d3
from holdon.preprocessing.homography import WallPose, camera_matrix

FOCAL, W, H = 600.0, 720, 1280


def _frontal_pose(dist: float = 2.0) -> WallPose:
    """Camera looking straight at the wall from ``dist`` m; wall origin at the image centre."""
    R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
    return WallPose(R=R, t=np.array([0.0, 0.0, dist]), K=camera_matrix(FOCAL, W, H), rms_px=0.0)


def _hold(i: int, x_m: float, y_m: float, half: float = 0.05, dist: float = 2.0) -> Hold:
    """Image box of a square hold centred at wall (x_m, y_m) seen by ``_frontal_pose``."""
    u = lambda x: W / 2 + FOCAL * x / dist  # noqa: E731
    v = lambda y: H / 2 - FOCAL * y / dist  # noqa: E731
    return Hold(str(i), round(u(x_m - half)), round(v(y_m + half)), round(u(x_m + half)), round(v(y_m - half)), False)


def test_point_polygon_distance_inside_edge_and_corner():
    square = np.array([[[0.0, 0], [1, 0], [1, 1], [0, 1]]])
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [2.0, 2.0], [0.5, -0.25]])
    d = d3.point_polygon_distance(pts, square)[:, 0]
    np.testing.assert_allclose(d, [0.0, 0.5, np.sqrt(2), 0.25])
    # Orientation of the polygon does not matter.
    np.testing.assert_allclose(d3.point_polygon_distance(pts, square[:, ::-1])[:, 0], d)


def test_limb_hold_distance_uses_closest_vertex_and_clamps_behind_wall():
    square = np.array([[[0.0, 0], [0.1, 0], [0.1, 0.1], [0.0, 0.1]]])
    pts = np.array([
        [[0.05, 0.05, 0.30], [0.05, 0.05, 0.02]],  # closest vertex 2 cm in front of the hold
        [[0.05, 0.05, -0.04], [1.0, 1.0, 1.0]],  # behind the wall plane, inside the outline -> 0
        [[np.nan] * 3, [np.nan] * 3],  # no body
    ])
    d = d3.limb_hold_distance(pts, square, side=1.0)[:, 0]
    assert d[0] == pytest.approx(0.02) and d[1] == 0.0 and np.isnan(d[2])


def _wham(points_cam: np.ndarray, focal: float = FOCAL) -> d3.WhamOutput:
    t = len(points_cam)
    return d3.WhamOutput("p0_green", focal, W, H, np.ones(t, bool),
                         {limb: points_cam for limb in LIMBS}, np.zeros(t, np.float32))


def test_compute_distance3d_end_to_end(tmp_path):
    pose = _frontal_pose()
    holds = [_hold(0, 0.0, 0.0), _hold(1, 0.5, 0.5)]
    # One vertex per frame, in wall coords: on hold 0; 10 cm in front of hold 0; next to hold 1.
    wall_pts = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.10]], [[0.5, 0.65, 0.0]]])
    cam_pts = wall_pts @ pose.R.T + pose.t
    dist = d3.compute_distance3d(_wham(cam_pts), pose, holds)
    assert dist.shape == (3, 4, 2) and dist.dtype == np.float32
    np.testing.assert_allclose(dist[:, 0, 0], [0.0, 0.10, np.hypot(0.45, 0.6)], atol=0.01)
    assert dist[2, 0, 1] == pytest.approx(0.10, abs=0.01)

    path = tmp_path / "p0_green.npz"
    d3.write_distance3d(path, "p0_green", dist, ["0", "1"])
    back = load_distance3d(path)
    assert back.method == "3d" and back.hold_ids == ("0", "1")
    np.testing.assert_array_equal(back.limb("left_hand"), dist[:, 0])


def test_compute_distance3d_refuses_a_different_focal_length():
    pose = _frontal_pose()
    with pytest.raises(ValueError, match="focal"):
        d3.compute_distance3d(_wham(np.zeros((1, 1, 3)), focal=1468.6), pose, [_hold(0, 0, 0)])
