"""Camera focal length and wall pose from The Way Up bolt holes (Track 3).

Each ``*_bolt_holes.json`` gives image points of reference bolt holes and their wall coordinates
(``z = 0``, centimetres; ``WALL_UNITS_PER_M``). With a pinhole camera (principal point at the image
centre, square pixels, no distortion) a plane seen at known points gives the wall's pose in the camera
frame, ``X_cam = R @ X_wall + t`` (OpenCV ``solvePnP``, IPPE planar solver, Levenberg-Marquardt refine).

**Bolt-hole order.** In 20 of the 22 files the two top points are listed in swapped order: the
image x of wall point ``(0, 510)`` is right of ``(135, 510)``, so the stored ``homography_mat`` maps
a twisted ("bow-tie") quadrilateral. ``fix_bolt_hole_order`` restores the order (image x must increase
with wall x along each row of points); after that all 22 videos fit a plane pose within ~2 px.

**Focal length.** The camera intrinsics are unknown. All recordings come from the same static camera
set-up, so one focal length is fitted jointly over all videos (``estimate_focal``). WHAM must run with
the same focal length so that body and wall live in the same camera frame (external/wham/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

WALL_UNITS_PER_M = 100.0
"""Bolt-hole wall coordinates are centimetres (x 0-135, y 0-510; docs/dataset_schema.md)."""


@dataclass(frozen=True)
class WallPose:
    """Pose of the wall plane in the camera frame: ``X_cam = R @ X_wall + t`` (metres)."""

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    K: np.ndarray  # (3, 3) camera matrix used
    rms_px: float  # bolt-hole reprojection error

    def cam_to_wall(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) to wall coordinates (..., 3) in metres; z = distance from the wall plane."""
        return (np.asarray(points, dtype=np.float64) - self.t) @ self.R

    def camera_side(self) -> float:
        """+1 or -1: sign of wall z on the camera's side of the wall."""
        return float(np.sign(self.cam_to_wall(np.zeros(3))[2]))


def camera_matrix(focal: float, width: int, height: int) -> np.ndarray:
    """Pinhole camera matrix with square pixels and the principal point at the image centre."""
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def fix_bolt_hole_order(img: np.ndarray, world: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return image points with swapped pairs restored, and whether anything was swapped.

    Points that share a wall y form a row; along each row the image x must increase with wall x
    (camera upright, facing the wall). Rows of exactly two points that violate this are swapped.
    Files with other layouts (e.g. ``p2a/green``, 350 points) are returned unchanged.
    """
    img = np.asarray(img, dtype=np.float64).copy()
    world = np.asarray(world, dtype=np.float64)
    swapped = False
    for y in np.unique(world[:, 1]):
        row = np.flatnonzero(world[:, 1] == y)
        if len(row) != 2:
            continue
        a, b = row[np.argsort(world[row, 0])]
        if img[b, 0] < img[a, 0]:
            img[[a, b]] = img[[b, a]]
            swapped = True
    return img, swapped


def wall_points_m(world: np.ndarray) -> np.ndarray:
    """Wall coordinates (N, 2 or 3) in centimetres to (N, 3) metres with z = 0."""
    xy = np.asarray(world, dtype=np.float64)[:, :2] / WALL_UNITS_PER_M
    return np.c_[xy, np.zeros(len(xy))]


def fit_wall_pose(img: np.ndarray, world: np.ndarray, focal: float, width: int, height: int) -> WallPose:
    """Wall pose from bolt holes (image pixels, wall centimetres) for a given focal length.

    Pass points already fixed with ``fix_bolt_hole_order``.
    """
    obj = wall_points_m(world)
    img = np.asarray(img, dtype=np.float64)
    K = camera_matrix(focal, width, height)
    # IPPE returns both planar solutions; refine each and keep the best one with the wall in front.
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
        if tvec[2, 0] <= 0:
            continue
        proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0][:, 0]
        rms = float(np.sqrt(((proj - img) ** 2).sum(axis=1).mean()))
        if best is None or rms < best.rms_px:
            best = WallPose(R=cv2.Rodrigues(rvec)[0], t=tvec[:, 0].copy(), K=K, rms_px=rms)
    if best is None:
        raise ValueError("no wall pose with the wall in front of the camera (bolt holes in the wrong order?)")
    return best


def recording_wall_pose(img: np.ndarray, world: np.ndarray, focal: float, width: int, height: int) -> tuple[WallPose, bool]:
    """Wall pose of one recording from its raw bolt holes (order fixed first); also returns whether a swap was fixed."""
    fixed, swapped = fix_bolt_hole_order(img, world)
    return fit_wall_pose(fixed, world, focal, width, height), swapped


def estimate_focal(
    walls: Iterable[tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
    candidates: Sequence[float] = tuple(np.arange(300.0, 2001.0, 5.0)),
) -> tuple[float, np.ndarray]:
    """One focal length for all videos: minimises the summed squared bolt-hole reprojection error.

    ``walls``: ``(img, world)`` pairs, already order-fixed. Returns ``(focal, total_sq_error)`` where the
    second item holds the objective for every candidate (for plotting / checking the minimum is interior).
    """
    walls = list(walls)
    cost = np.zeros(len(candidates))
    for i, f in enumerate(candidates):
        for img, world in walls:
            cost[i] += fit_wall_pose(img, world, f, width, height).rms_px ** 2
    return float(candidates[int(np.argmin(cost))]), cost


def image_to_wall_homography(pose: WallPose) -> np.ndarray:
    """Homography mapping image pixels to wall (x, y) in metres, implied by the wall pose."""
    wall_to_img = pose.K @ np.c_[pose.R[:, :2], pose.t]
    H = np.linalg.inv(wall_to_img)
    return H / H[2, 2]


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (..., 2) points with a 3x3 homography."""
    pts = np.asarray(points, dtype=np.float64)
    h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1) @ H.T
    return h[..., :2] / h[..., 2:3]
