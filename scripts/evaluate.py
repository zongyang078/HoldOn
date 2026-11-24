"""Score an event file against The Way Up ground truth and write metric tables.

Prints and writes (next to the event file, ``metrics.md`` by default) the macro P / R / F1 at every
tIoU threshold of ``configs/eval.yaml`` for all limbs, hands, feet, each route, occluded vs visible
contact frames and climber-experience groups (group averages only). Frame counts are read from the
videos with ``holdon.data.video.video_info``.

Usage:
    python3 scripts/evaluate.py results/track1/<run>/events_lopo.csv [--out metrics.md]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.video import video_info
from holdon.evaluation.grouped import experience_groups, load_experience
from holdon.evaluation.report import breakdown_report, load_ground_truth
from holdon.events import read_events


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("events", help="events CSV (docs/event_format.md) with predictions for all videos")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--out", default="metrics.md", help="file name written next to the events file")
    parser.add_argument("--title", help="report title (default: the run folder name)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)  # known label quirks

    eval_cfg = load_config(args.eval_config)
    paths = load_config(args.paths)
    root = Path(paths["raw"]["wayup"])
    events_path = Path(args.events)

    labeled = load_ground_truth(
        root, lambda rec, route: video_info(wayup.video_path(root, rec, route)).num_frames)
    preds: dict[str, list] = {v: [] for v in labeled}
    for e in read_events(events_path):
        if e.video_id not in preds:
            raise SystemExit(f"{events_path}: unknown video {e.video_id!r}")
        preds[e.video_id].append(e)

    exp_cfg = eval_cfg.get("experience")
    groups = None
    if exp_cfg:
        groups = experience_groups(load_experience(root / exp_cfg["form"], exp_cfg["column"]),
                                   exp_cfg["min_participants"])

    lines = breakdown_report(preds, labeled, eval_cfg["tiou_thresholds"],
                             title=args.title or f"Evaluation — {events_path.parent.name}/{events_path.name}",
                             experience_groups=groups)
    out = events_path.parent / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
