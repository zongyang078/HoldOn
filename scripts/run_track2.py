"""Run Track 2 (learned per-frame contact classifier) with LOPO training and report held-out metrics.

Builds the feature table of every video, trains and tunes fold by fold
(``holdon.training.train.lopo_train_predict``) and writes to
``<results>/track2/<pose_version>_<run_name>/``:

- ``events_lopo.csv``: held-out predictions of every video,
- ``config.yaml``: the config used,
- ``chosen_params.csv``: decoding parameters chosen in each fold (on training participants only),
- ``candidates.md``: how many ground-truth contact frames the candidate selection keeps,
- ``metrics.md``: the ``scripts/evaluate.py`` report of ``events_lopo.csv``.

Usage:
    python3 scripts/run_track2.py [--config configs/track2.yaml] [--pose-version ...] [--run-name ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from holdon.config import load_config
from holdon.training.pipeline import run_track2


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/track2.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--pose-version", help="override pose_version from the config")
    parser.add_argument("--run-name", help="override run_name from the config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)  # known label quirks

    cfg = load_config(args.config)
    if args.pose_version:
        cfg["pose_version"] = args.pose_version
    if args.run_name:
        cfg["run_name"] = args.run_name
    out_dir = run_track2(cfg, load_config(args.eval_config), load_config(args.paths))
    print((Path(out_dir) / "metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
