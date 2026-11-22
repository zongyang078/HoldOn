"""Fit one camera focal length to the bolt holes of all videos and report each video's wall pose.

Writes ``results/tables/track3_wall_pose.md``: the shared focal length, the reprojection error with it
and with WHAM's default focal length, which files had swapped bolt holes, the camera position in wall
coordinates, and hold box sizes on the wall (a check of the centimetre units). Put the focal length
into ``configs/track3.yaml`` (``focal_length_px``) by hand when it changes.

Usage:
    python3 scripts/wall_pose.py [--paths configs/paths.yaml] [--out results/tables/track3_wall_pose.md]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.video import video_info
from holdon.preprocessing import homography as hg


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--out", default="results/tables/track3_wall_pose.md")
    args = parser.parse_args()
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)

    root = Path(load_config(args.paths)["raw"]["wayup"])
    recs = []
    for rec in wayup.list_recordings(root):
        for route in wayup.ROUTES:
            info = video_info(wayup.video_path(root, rec, route))
            wall = wayup.load_wall_reference(root / rec / f"{route}_bolt_holes.json")
            holds = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
            recs.append((wayup.video_id(rec, route), info, wall, holds))
    sizes = {(i.width, i.height) for _, i, _, _ in recs}
    if len(sizes) != 1:
        raise SystemExit(f"videos differ in size: {sizes}")
    width, height = sizes.pop()

    fixed = [hg.fix_bolt_hole_order(w.img_coords, w.world_coords)[0] for _, _, w, _ in recs]
    focal, cost = hg.estimate_focal(((img, w.world_coords) for img, (_, _, w, _) in zip(fixed, recs)), width, height)
    wham_focal = float(np.hypot(width, height))

    rows, hold_sizes = [], []
    for vid, _, wall, holds in recs:
        pose, swapped = hg.recording_wall_pose(wall.img_coords, wall.world_coords, focal, width, height)
        wham_rms = hg.recording_wall_pose(wall.img_coords, wall.world_coords, wham_focal, width, height)[0].rms_px
        cam = pose.cam_to_wall(np.zeros(3))
        H = hg.image_to_wall_homography(pose)
        for h in holds:
            a, b = hg.apply_homography(H, np.array([[h.x1, h.y1], [h.x2, h.y2]], dtype=float))
            hold_sizes.append(np.abs(b - a))
        rows.append(f"| {vid} | {len(wall.img_coords)} | {'yes' if swapped else 'no'} | {pose.rms_px:.2f} | "
                    f"{wham_rms:.2f} | {cam[0]:.2f}, {cam[1]:.2f}, {abs(cam[2]):.2f} |")
    hold_sizes = np.array(hold_sizes) * 100
    rms = [float(r.split("|")[4]) for r in rows]

    text = f"""# Track 3 wall pose (bolt holes)

Made by `scripts/wall_pose.py`. Pinhole camera, principal point at the image centre ({width}×{height}),
one focal length for all 22 videos fitted to the bolt holes (`holdon.preprocessing.homography`).

- **Shared focal length: {focal:.0f} px** (WHAM default sqrt(w² + h²) = {wham_focal:.1f} px).
  Summed squared bolt-hole error: {cost.min():.1f} px² at {focal:.0f} px vs
  {cost[int(np.argmin(np.abs(np.arange(300.0, 2001.0, 5.0) - wham_focal)))]:.1f} px² at the WHAM default.
- Bolt-hole RMS error with the shared focal: median {np.median(rms):.2f} px, max {max(rms):.2f} px.
- {sum('| yes |' in r for r in rows)} of {len(rows)} files list the two top bolt holes in swapped order
  (fixed by `fix_bolt_hole_order`; the stored `homography_mat` of these files is wrong).
- Hold boxes mapped to the wall: median {np.median(hold_sizes[:, 0]):.1f} × {np.median(hold_sizes[:, 1]):.1f} cm,
  90th percentile {np.percentile(hold_sizes[:, 0], 90):.1f} × {np.percentile(hold_sizes[:, 1], 90):.1f} cm
  (plausible hold sizes, so wall units are centimetres).

| Video | Bolt holes | Swapped | RMS px (shared f) | RMS px (WHAM f) | Camera in wall coords x, y, distance (m) |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
