"""3D limb-to-hold distances (core of Track 3, input to Track 2).

Inputs: the WHAM files of the 3D line (external/wham/README.md: hand and foot SMPL vertices in the
camera frame, metres) and the wall pose from the bolt holes (``holdon.preprocessing.homography``),
both with the same focal length. Output: the distance files of docs/interfaces.md section 6.

Model (recorded in each version's MANIFEST):

- **Limb:** its SMPL contact vertices (palm and fingers; sole, heel and toes). The limb-to-hold distance
  is the smallest distance over these vertices.
- **Hold:** its image box mapped onto the wall plane through the wall pose: a flat quadrilateral at
  wall ``z = 0``.
- **Distance:** Euclidean distance from a vertex to that quadrilateral. Points behind the wall plane
  (seen from the camera) count as being at the wall, so the distance is 0 when a vertex is inside the
  hold's outline at or behind the wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from holdon.data.external import DISTANCE3D_FORMAT_VERSION
from holdon.data.wayup import Hold
from holdon.events import LIMBS
from holdon.preprocessing.homography import WallPose, apply_homography, image_to_wall_homography

WHAM_FORMAT_VERSION = 1


@dataclass(frozen=True)
class WhamOutput:
    """The parts of one WHAM file (external/wham/run_wham.py) that Track 3 uses."""

    video_id: str
    focal_length: float
    width: int
    height: int
    valid: np.ndarray  # (T,) bool
    limb_points: dict[str, np.ndarray]  # limb -> (T, V_limb, 3) camera frame, metres, NaN if not valid
    reproj_err_px: np.ndarray  # (T,)

    @property
    def num_frames(self) -> int:
        """Number of video frames covered."""
        return int(self.valid.shape[0])


def load_wham(path: str | Path) -> WhamOutput:
    """Load one ``<video_id>.npz`` written by external/wham/run_wham.py."""
    with np.load(Path(path), allow_pickle=False) as d:
        version = int(d["format_version"])
        if version != WHAM_FORMAT_VERSION:
            raise ValueError(f"{path}: WHAM format_version {version}, expected {WHAM_FORMAT_VERSION}")
        limbs = tuple(str(x) for x in d["limbs"])
        if limbs != LIMBS:
            raise ValueError(f"{path}: limbs {limbs}, expected {LIMBS}")
        verts = d["limb_verts"].astype(np.float64)
        bounds = np.r_[0, np.cumsum(d["limb_vertex_counts"])]
        points = {limb: verts[:, bounds[i]:bounds[i + 1]] for i, limb in enumerate(LIMBS)}
        return WhamOutput(
            video_id=str(d["video_id"]),
            focal_length=float(d["focal_length"]),
            width=int(d["width"]),
            height=int(d["height"]),
            valid=d["valid"].astype(bool),
            limb_points=points,
            reproj_err_px=d["reproj_err_px"].astype(np.float32),
        )


def hold_polygons(holds: Sequence[Hold], img_to_wall: np.ndarray) -> np.ndarray:
    """(H, 4, 2) hold boxes mapped to wall (x, y) metres; corners in image order TL, TR, BR, BL."""
    corners = np.array([[(h.x1, h.y1), (h.x2, h.y1), (h.x2, h.y2), (h.x1, h.y2)] for h in holds], dtype=np.float64)
    return apply_homography(img_to_wall, corners)


def point_polygon_distance(points: np.ndarray, polygons: np.ndarray) -> np.ndarray:
    """In-plane distance from points (N, 2) to convex polygons (H, C, 2); 0 inside. Returns (N, H)."""
    p = points[:, None, None, :]  # (N, 1, 1, 2)
    a = polygons[None]  # (1, H, C, 2)
    b = np.roll(polygons, -1, axis=1)[None]
    ab = b - a
    ap = p - a
    u = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    edge_dist = np.linalg.norm(ap - u[..., None] * ab, axis=-1).min(axis=-1)  # (N, H)
    cross = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]  # (N, H, C)
    inside = (cross >= 0).all(-1) | (cross <= 0).all(-1)
    return np.where(inside, 0.0, edge_dist)


def limb_hold_distance(points_wall: np.ndarray, polygons: np.ndarray, side: float, chunk: int = 128) -> np.ndarray:
    """(T, H) smallest distance from a limb's vertices (T, V, 3) in wall coordinates to each hold.

    ``side`` is the sign of wall z on the camera's side; points on the far side count as z = 0.
    Frames whose vertices are NaN give NaN.
    """
    t_len, v_len, _ = points_wall.shape
    out = np.full((t_len, len(polygons)), np.nan)
    for s in range(0, t_len, chunk):
        block = points_wall[s:s + chunk]
        ok = ~np.isnan(block).any(axis=(1, 2))
        if not ok.any():
            continue
        pts = block[ok].reshape(-1, 3)
        dxy = point_polygon_distance(pts[:, :2], polygons)  # (n*V, H)
        dz = np.maximum(side * pts[:, 2], 0.0)[:, None]
        d = np.sqrt(dxy ** 2 + dz ** 2).reshape(-1, v_len, len(polygons)).min(axis=1)
        out[s:s + chunk][ok] = d
    return out


def compute_distance3d(wham: WhamOutput, pose: WallPose, holds: Sequence[Hold]) -> np.ndarray:
    """(T, 4, H) limb-to-hold distances in metres for one video (NaN where WHAM has no body)."""
    if abs(pose.K[0, 0] - wham.focal_length) > 0.5:
        raise ValueError(
            f"{wham.video_id}: WHAM ran with focal {wham.focal_length:.1f} px, wall pose uses "
            f"{pose.K[0, 0]:.1f} px; rerun one of them so both use the same camera"
        )
    polygons = hold_polygons(holds, image_to_wall_homography(pose))
    side = pose.camera_side()
    dist = np.stack(
        [limb_hold_distance(pose.cam_to_wall(wham.limb_points[limb]), polygons, side) for limb in LIMBS], axis=1
    )
    return dist.astype(np.float32)


def write_distance3d(path: str | Path, video_id: str, dist: np.ndarray, hold_ids: Sequence[str],
                     method: str = "3d") -> None:
    """Write one distance file in the format of docs/interfaces.md section 6 (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        dist=np.asarray(dist, dtype=np.float32),
        limbs=np.array(LIMBS),
        hold_ids=np.array([str(h) for h in hold_ids]),
        video_id=np.array(video_id),
        method=np.array(method),
        format_version=np.int32(DISTANCE3D_FORMAT_VERSION),
    )
    tmp.replace(path)
