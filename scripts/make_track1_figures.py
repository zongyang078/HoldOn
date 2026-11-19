"""Make the Phase 2 figures for Track 1: overlay frame, contact timeline, false-positive example.

Reads the LOPO predictions written by ``scripts/tune_track1.py`` and writes PNG/JPG files to
``<results>/figures/``. Optionally also a before/after image of the climber-selection fix.

Usage:
    python3 scripts/tune_track1.py --keypoints fingertip_toe      # first, once
    python3 scripts/make_track1_figures.py [--video p8_orange] [--keypoints fingertip_toe]
    python3 scripts/make_track1_figures.py --v1-pose-dir <old v1 folder>   # adds the selection-fix image
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.data.video import read_frame
from holdon.events import read_events
from holdon.features.overlap2d import min_distance
from holdon.models.rules import LIMB_KEYPOINTS
from holdon.models.track1_tuning import match_events
from holdon.visualization.overlay import (
    GT_COLOR, PRED_COLOR, draw_holds, draw_keypoints, highlight_holds,
)
from holdon.visualization.timeline import plot_timeline

HANDS = ("left_hand", "right_hand")


def active(events, t):
    """Events whose interval contains frame t."""
    return [e for e in events if e.start_frame <= t <= e.end_frame]


def crop_around(frame, box, margin=120, min_h=520):
    """Crop a frame around a box (x1, y1, x2, y2), at least ``min_h`` pixels high."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cy, half = (y1 + y2) / 2, max((y2 - y1) / 2 + margin, min_h / 2)
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    x1, x2 = int(max(0, x1 - margin)), int(min(w, x2 + margin))
    return frame[y1:y2, x1:x2]


def caption(img, lines):
    """Add a white band above the image with one or more lines of black text.

    The image is padded with white on the right if the text is wider than the image.
    """
    font, scale, step = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 20
    width = max(img.shape[1], *(cv2.getTextSize(t, font, scale, 1)[0][0] + 12 for t in lines))
    if width > img.shape[1]:
        img = np.hstack([img, np.full((img.shape[0], width - img.shape[1], 3), 255, np.uint8)])
    band = np.full((step * len(lines) + 8, width, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(band, text, (6, step * (i + 1)), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def render(video, pose, holds, gt, pred, t, points, lines):
    """Frame t with all holds, GT holds (orange), predicted holds (magenta) and keypoints, cropped."""
    frame = draw_holds(read_frame(video, t), holds, label=False)
    frame = highlight_holds(frame, holds, [e.hold_id for e in active(gt, t)], GT_COLOR, 3, pad=3)
    frame = highlight_holds(frame, holds, [e.hold_id for e in active(pred, t)], PRED_COLOR, 3, pad=9)
    frame = draw_keypoints(frame, [tuple(pose.track(n)[0][t]) for n in points])
    return caption(crop_around(frame, pose.bbox[t]), lines)


LEGEND = "orange: labeled contact | magenta: Track 1 | red: keypoints | green/blue: hand/foot holds"


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--video", default="p8_orange")
    parser.add_argument("--keypoints", default="fingertip_toe")
    parser.add_argument("--v1-pose-dir", type=Path, help="v1 pose folder for the selection-fix figure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)

    paths = load_config(args.paths)
    root = Path(paths["raw"]["wayup"])
    version = paths["versions"]["pose2d"]
    run_dir = Path(paths["results"]) / "track1" / f"{version}_{args.keypoints}"
    out_dir = Path(paths["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pred = read_events(run_dir / "events_lopo.csv")
    points = [n for limb in LIMB_KEYPOINTS[args.keypoints].values() for n in limb]

    def load(vid):
        rec, route = vid.rsplit("_", 1)
        pose = load_pose2d(pose2d_path(Path(paths["interim"]["pose2d"]) / version, vid))
        r = wayup.load_recording(root, rec, route, pose.num_frames)
        return r, pose, [le.event for le in r.events], [e for e in all_pred if e.video_id == vid]

    # 1. Overlay: the frame of the chosen video with the most limbs in correct predicted contact.
    r, pose, gt, pred = load(args.video)
    def correct(t):
        g = {(e.limb, e.hold_id) for e in active(gt, t)}
        return len(g & {(e.limb, e.hold_id) for e in active(pred, t)})
    t = max(range(0, pose.num_frames, 5), key=lambda i: (correct(i), -abs(i - pose.num_frames / 2)))
    img = render(r.video, pose, r.holds, gt, pred, t, points,
                 [f"{args.video}, t = {t / pose.fps:.1f} s, Track 1 ({args.keypoints})", LEGEND])
    cv2.imwrite(str(out_dir / f"track1_overlay_{args.video}.jpg"), img)

    # 2. Timeline of the same video.
    fig = plot_timeline(gt, pred, pose.fps, title=f"{args.video}: ground truth vs Track 1 ({args.keypoints})")
    fig.savefig(out_dir / f"track1_timeline_{args.video}.png", dpi=150)

    # 3. False positive: an unmatched hand prediction (tIoU > 0). Shown at the frame where the hand's
    #    keypoints are closest to the predicted hold; preferred: no labeled contact of that hand at that
    #    moment (the hand is moving past the hold), and the keypoints inside the hold box.
    candidates = []
    for vid in sorted({e.video_id for e in all_pred}):
        r2, pose2, gt2, pred2 = load(vid)
        matched = {i for i, _, _ in match_events(pred2, gt2, 0.0)}
        boxes = {h.hold_id: np.array([[h.x1, h.y1, h.x2, h.y2]], float) for h in r2.holds}
        for i, e in enumerate(pred2):
            if i in matched or e.limb not in HANDS:
                continue
            frames = np.arange(e.start_frame, e.end_frame + 1)
            tracks = [pose2.track(n)[0][frames] for n in LIMB_KEYPOINTS[args.keypoints][e.limb]]
            dist = min_distance(tracks, boxes[e.hold_id])[:, 0]
            if np.isnan(dist).all():
                continue
            best = int(np.nanargmin(dist))
            t_best = int(frames[best])
            free = not any(g.limb == e.limb for g in active(gt2, t_best))
            candidates.append((free, -dist[best], e.score, vid, e, t_best))
    if candidates:
        *_, vid, e, mid = max(candidates, key=lambda c: c[:3])
        r2, pose2, gt2, _ = load(vid)
        limb_gt = [g for g in gt2 if g.limb == e.limb]
        img = render(r2.video, pose2, r2.holds, limb_gt, [e], mid, LIMB_KEYPOINTS[args.keypoints][e.limb],
                     [f"False positive ({vid}, t = {mid / pose2.fps:.1f} s): {e.limb.replace('_', ' ')} "
                      f"predicted on hold {e.hold_id} for {e.num_frames / pose2.fps:.1f} s, not labeled",
                      "magenta: predicted hold | orange: labeled contact of this hand | red: its fingertips"])
        cv2.imwrite(str(out_dir / "track1_false_positive.jpg"), img)
        logging.info("false positive example: %s %s", vid, e)

    # 4. Climber selection before / after the route-region fix.
    if args.v1_pose_dir:
        vid, t = "p1_green", 3931
        rec, route = vid.rsplit("_", 1)
        old = np.load(pose2d_path(args.v1_pose_dir, vid))  # format v1: load_pose2d rejects it on purpose
        new = load_pose2d(pose2d_path(Path(paths["interim"]["pose2d"]) / version, vid))
        holds = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
        frame = draw_holds(read_frame(wayup.video_path(root, rec, route), t), holds, label=False)
        panels = []
        for name, box, color in (("before: belayer tracked", old["bbox"][t], (0, 0, 255)),
                                 ("after: route-region filter", new.bbox[t], (0, 200, 0))):
            p = frame.copy()
            x1, y1, x2, y2 = np.nan_to_num(box).astype(int)
            cv2.rectangle(p, (x1, y1), (x2, y2), color, 4)
            panels.append(caption(cv2.resize(p, None, fx=0.5, fy=0.5), [name]))
        cv2.imwrite(str(out_dir / "pose_selection_fix.jpg"), np.hstack(panels))

    logging.info("figures -> %s", out_dir)


if __name__ == "__main__":
    main()
