"""Run Track 1 (2D keypoint-in-hold-box baseline) on every video and write contact events.

Reads the pose version named in the Track 1 config and writes
``<results>/track1/<pose_version>_<keypoints>/events.csv`` plus the config used.

Usage:
    python3 scripts/run_track1.py [--config configs/track1.yaml] [--paths configs/paths.yaml]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.events import write_events
from holdon.models.rules import track1_events


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/track1.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--only", nargs="*", help="video ids, e.g. p1_green")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    paths = load_config(args.paths)
    cfg["pose_version"] = cfg.get("pose_version") or paths["versions"]["pose2d"]
    root = Path(paths["raw"]["wayup"])
    pose_dir = Path(paths["interim"]["pose2d"]) / cfg["pose_version"]
    out_dir = Path(paths["results"]) / "track1" / f"{cfg['pose_version']}_{cfg['keypoints']}"

    events = []
    for rec in wayup.list_recordings(root):
        for route in wayup.ROUTES:
            vid = wayup.video_id(rec, route)
            if args.only and vid not in args.only:
                continue
            pose = load_pose2d(pose2d_path(pose_dir, vid))
            holds = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
            found = track1_events(pose, holds, cfg)
            logging.info("%s: %d events", vid, len(found))
            events += found

    write_events(events, out_dir / "events.csv")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    logging.info("%d events -> %s", len(events), out_dir / "events.csv")


if __name__ == "__main__":
    main()
