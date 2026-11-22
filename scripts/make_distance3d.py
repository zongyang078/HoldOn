"""Turn WHAM files into the limb-to-hold distance files of docs/interfaces.md section 6.

Reads ``<wham-dir>/<video_id>.npz`` (external/wham/run_wham.py), fits each video's wall pose with the
focal length of ``configs/track3.yaml`` (which WHAM must have used too), and writes
``<distance3d>/<name>_<date>/<video_id>.npz`` plus ``MANIFEST.md``. Every written file is read back with
``holdon.data.external.load_distance3d``. Prints, per video, how far the labeled limb is from its labeled
hold during labeled contacts compared with the other frames (a first check that the distances separate).

Usage:
    python3 scripts/make_distance3d.py --wham-dir data/interim/wham/wham-vitposebox_2026-11-22 [--only p3_orange]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
from pathlib import Path

import numpy as np

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import distance3d_path, load_distance3d
from holdon.events import LIMBS
from holdon.features.distance3d import compute_distance3d, load_wham, write_distance3d
from holdon.preprocessing.homography import recording_wall_pose


def contact_summary(dist: np.ndarray, hold_ids: list[str], events) -> tuple[float, float]:
    """Median distance (m) of the labeled limb to its labeled hold inside vs outside labeled contacts."""
    inside, outside = [], []
    for li, limb in enumerate(LIMBS):
        for hi, hold in enumerate(hold_ids):
            mask = np.zeros(len(dist), bool)
            for ev in events:
                e = ev.event
                if e.limb == limb and e.hold_id == hold:
                    mask[e.start_frame:e.end_frame + 1] = True
            if mask.any():
                d = dist[:, li, hi]
                inside.append(d[mask])
                outside.append(d[~mask])
    cat = lambda xs: np.concatenate(xs) if xs else np.array([np.nan])  # noqa: E731
    return float(np.nanmedian(cat(inside))), float(np.nanmedian(cat(outside)))


def git_commit() -> str:
    """Short commit hash of the working tree (``unknown`` outside git)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wham-dir", type=Path, required=True, help="WHAM version folder")
    parser.add_argument("--config", default="configs/track3.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--out", type=Path, help="parent folder (default: paths.yaml processed.distance3d)")
    parser.add_argument("--name", default="wham3d")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--only", nargs="*", help="video ids (default: every WHAM file)")
    args = parser.parse_args()
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)

    cfg, paths = load_config(args.config), load_config(args.paths)
    focal = cfg["focal_length_px"]
    if focal is None:
        raise SystemExit(f"set focal_length_px in {args.config} (scripts/wall_pose.py)")
    root = Path(paths["raw"]["wayup"])
    out_dir = (args.out or Path(paths["processed"]["distance3d"])) / f"{args.name}_{args.date}"
    files = sorted(args.wham_dir.glob("*.npz"))
    if args.only:
        files = [f for f in files if f.stem in args.only]
    if not files:
        raise SystemExit(f"no WHAM files in {args.wham_dir}")

    lines = []
    for f in files:
        wham = load_wham(f)
        rec, route = wham.video_id.rsplit("_", 1)
        recording = wayup.load_recording(root, rec, route, num_frames=wham.num_frames)
        pose, _ = recording_wall_pose(recording.wall.img_coords, recording.wall.world_coords, focal,
                                      wham.width, wham.height)
        hold_ids = [h.hold_id for h in recording.holds]
        if hold_ids != [str(i) for i in range(len(hold_ids))]:
            raise SystemExit(f"{wham.video_id}: hold ids are not 0..H-1 in file order: {hold_ids}")
        dist = compute_distance3d(wham, pose, recording.holds)
        path = distance3d_path(out_dir, wham.video_id)
        write_distance3d(path, wham.video_id, dist, hold_ids, method="3d")
        check = load_distance3d(path)
        assert check.num_frames == wham.num_frames and check.hold_ids == tuple(hold_ids)
        d_in, d_out = contact_summary(dist, hold_ids, recording.events)
        line = (f"{wham.video_id:12s} {wham.num_frames:5d} frames, body {wham.valid.mean():6.1%}, "
                f"labeled limb-hold distance: in contact {d_in:.3f} m, otherwise {d_out:.3f} m")
        print(line, flush=True)
        lines.append(line)

    manifest = out_dir / "MANIFEST.md"
    if not manifest.exists():
        manifest.write_text(f"""# MANIFEST — {out_dir.name}

| Field | Value |
| --- | --- |
| Created by | <fill in> |
| Date | {dt.date.today().isoformat()} |
| Code commit | {git_commit()} |
| Producer | scripts/make_distance3d.py |
| Model / weights | WHAM output `{args.wham_dir.name}` (external/wham/run_wham.py) |
| Config | `{args.config}`: focal_length_px {focal} |
| Input | `{args.wham_dir.name}`, bolt holes and hold boxes of `wayup.zip` |
| Environment | <fill in> |

## Files

`<video_id>.npz`: docs/interfaces.md section 6, method `3d`.

- Limb: closest SMPL contact vertex (palm and fingers; sole, heel, toes; external/wham/README.md).
- Hold: image box mapped onto the wall plane (flat quadrilateral at wall z = 0).
- Wall pose: bolt holes (order fixed), solvePnP with focal {focal} px, principal point at the image centre.
- Distance: 3D point-to-quadrilateral distance; points behind the wall plane count as on it. No smoothing.
- NaN: frames without a WHAM body.

## Notes

<fill in>

## Runs
""")
    with manifest.open("a") as fh:
        fh.writelines(f"    {line}\n" for line in lines)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
