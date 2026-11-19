"""Render a Track 1 demo video: labeled vs predicted contacts drawn on the climbing video.

Reads the LOPO predictions written by ``scripts/tune_track1.py`` (held-out predictions only) and
writes ``<results>/videos/track1_<video>_<start>-<end>s.mp4``. Each frame shows all hold boxes, the
labeled contact holds (orange), the Track 1 contact holds (magenta), the limb keypoints (red), and a
status band with labeled vs predicted hold per limb (green row = agree, red row = disagree).

Usage:
    python3 scripts/make_overlay.py --video p8_orange --start 10 --end 40
    python3 scripts/make_overlay.py --video p2a_green --start 17 --end 27 --scale 0.75

OpenCV writes MPEG-4 Part 2 (``mp4v``). If ``ffmpeg`` is on the PATH the file is re-encoded to H.264,
which PowerPoint, Keynote and Google Slides play reliably; ``--no-h264`` skips that step.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.data.video import iter_frames
from holdon.events import LIMBS, read_events
from holdon.models.rules import LIMB_KEYPOINTS
from holdon.visualization.overlay import (
    GT_COLOR, PRED_COLOR, contact_panel, draw_holds, draw_keypoints, highlight_holds,
)


def at(events, t):
    """limb -> hold id of the event active at frame t (None if no event)."""
    out = {limb: None for limb in LIMBS}
    for e in events:
        if e.start_frame <= t <= e.end_frame:
            out[e.limb] = e.hold_id
    return out


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--video", default="p8_orange")
    parser.add_argument("--keypoints", default="fingertip_toe")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="end time in seconds (default: video end)")
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor of the output video")
    parser.add_argument("--no-h264", action="store_true", help="keep OpenCV's mp4v file, skip ffmpeg")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = load_config(args.paths)
    version = paths["versions"]["pose2d"]
    run_dir = Path(paths["results"]) / "track1" / f"{version}_{args.keypoints}"
    rec, route = args.video.rsplit("_", 1)
    pose = load_pose2d(pose2d_path(Path(paths["interim"]["pose2d"]) / version, args.video))
    r = wayup.load_recording(Path(paths["raw"]["wayup"]), rec, route, pose.num_frames)
    gt = [le.event for le in r.events]
    pred = [e for e in read_events(run_dir / "events_lopo.csv") if e.video_id == args.video]
    tracks = {limb: [pose.track(n)[0] for n in names] for limb, names in LIMB_KEYPOINTS[args.keypoints].items()}

    first = int(round(args.start * pose.fps))
    last = pose.num_frames - 1 if args.end is None else min(pose.num_frames - 1, int(round(args.end * pose.fps)))
    end_s = (last + 1) / pose.fps
    out_dir = Path(paths["results"]) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"track1_{args.video}_{args.start:g}-{end_s:.0f}s.mp4"

    writer = None
    for t, frame in iter_frames(r.video):
        if t < first:
            continue
        if t > last:
            break
        g, p = at(gt, t), at(pred, t)
        img = draw_holds(frame, r.holds, label=False)
        img = highlight_holds(img, r.holds, [h for h in g.values() if h], GT_COLOR, 3, pad=3)
        img = highlight_holds(img, r.holds, [h for h in p.values() if h], PRED_COLOR, 3, pad=9)
        img = draw_keypoints(img, [tuple(tr[t]) for trs in tracks.values() for tr in trs])
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        header = f"{args.video}  t = {t / pose.fps:5.1f} s   orange: labeled  magenta: Track 1"
        img = np.vstack([contact_panel(img.shape[1], g, p, header), img])
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (w, h))
        writer.write(img)
    if writer is None:
        raise SystemExit(f"No frames in [{args.start}, {args.end}] s")
    writer.release()
    if not args.no_h264 and shutil.which("ffmpeg"):
        tmp = out_path.with_suffix(".h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", str(tmp)], check=True)
        tmp.replace(out_path)
    logging.info("wrote %s (%d frames)", out_path, last - first + 1)


if __name__ == "__main__":
    main()
