"""Draw hold boxes on the first and last frame of each processed video to check alignment.

Writes one side-by-side image per video to ``<results>/alignment/``.

Usage:
    python3 scripts/check_alignment.py [--paths configs/paths.yaml]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.video import read_frame, video_info
from holdon.visualization.overlay import draw_holds


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = load_config(args.paths)
    root = Path(paths["raw"]["wayup"])
    out_dir = Path(paths["results"]) / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in wayup.list_recordings(root):
        for route in wayup.ROUTES:
            video = wayup.video_path(root, rec, route)
            info = video_info(video)
            holds = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
            first = draw_holds(read_frame(video, 0), holds)
            last = draw_holds(read_frame(video, info.num_frames - 1), holds)
            out = out_dir / f"{wayup.video_id(rec, route)}.jpg"
            cv2.imwrite(str(out), np.hstack([first, last]))
            logging.info("%s: %d holds, %d frames -> %s", out.stem, len(holds), info.num_frames, out)


if __name__ == "__main__":
    main()
