"""Make the Phase 2 figures for Track 2: contact timelines (ground truth vs Track 2 v1 held-out events).

Writes results/figures/track2_timeline_<video>.png for each --video and prints, for the Track 1
false-positive example (p2a_green, right hand, hold 11), whether Track 2 predicts that contact.

    python scripts/make_track2_figures.py --run rtmw-x-384_2026-11-18_v1 --video p8_orange p2a_green
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.events import read_events
from holdon.visualization.timeline import plot_timeline


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--run", default="rtmw-x-384_2026-11-18_v1", help="folder under results/track2/")
    parser.add_argument("--video", nargs="+", default=["p8_orange", "p2a_green"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)

    paths = load_config(args.paths)
    root = Path(paths["raw"]["wayup"])
    version = paths["versions"]["pose2d"]
    out_dir = Path(paths["results"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pred = read_events(Path(paths["results"]) / "track2" / args.run / "events_lopo.csv")

    for vid in args.video:
        rec, route = vid.rsplit("_", 1)
        pose = load_pose2d(pose2d_path(Path(paths["interim"]["pose2d"]) / version, vid))
        gt = [le.event for le in wayup.load_recording(root, rec, route, pose.num_frames).events]
        pred = [e for e in all_pred if e.video_id == vid]
        fig = plot_timeline(gt, pred, pose.fps, title=f"{vid}: ground truth vs Track 2 v1 (SVM)")
        out = out_dir / f"track2_timeline_{vid}.png"
        fig.savefig(out, dpi=150)
        logging.info("wrote %s", out)
        if vid == "p2a_green":
            hits = [e for e in pred if e.limb == "right_hand" and e.hold_id == "11"]
            for e in hits:
                print(f"p2a_green right_hand hold 11: {e.start_frame / pose.fps:.1f}-{e.end_frame / pose.fps:.1f} s")
            labeled = [e for e in gt if e.limb == "right_hand" and e.hold_id == "11"]
            for e in labeled:
                print(f"  labeled: {e.start_frame / pose.fps:.1f}-{e.end_frame / pose.fps:.1f} s")
            if not hits:
                print("p2a_green right_hand hold 11: no Track 2 contact")


if __name__ == "__main__":
    main()
