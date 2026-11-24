"""Tune Track 1 thresholds with leave-one-participant-out CV and report held-out metrics.

For each fold the grid in ``configs/track1.yaml`` (``tuning``) is scored on the training
participants only; the best config is applied to the held-out participant. Writes to
``<results>/track1/<pose_version>_<keypoints>/``:

- ``events_lopo.csv``: held-out predictions of every video,
- ``metrics.md``: P / R / F1 / mean tIoU (macro over videos), all limbs, hands, feet, plus the
  untuned default config for reference,
- ``chosen_params.csv``: the config picked in each fold.

Usage:
    python3 scripts/tune_track1.py [--config configs/track1.yaml] [--keypoints fingertip_toe]
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from holdon.config import load_config
from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.events import write_events
from holdon.models.rules import track1_events
from holdon.evaluation.metrics import FEET, HANDS, macro_metrics, video_counts
from holdon.models.track1_tuning import expand_grid, lopo_tune


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/track1.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--pose-version", help="override pose_version from the config")
    parser.add_argument("--keypoints", help="override keypoints (wrist_ankle | fingertip_toe)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("holdon.data.wayup").setLevel(logging.ERROR)  # known label quirks

    cfg = load_config(args.config)
    cfg["keypoints"] = args.keypoints or cfg["keypoints"]
    thresholds = [float(t) for t in load_config(args.eval_config)["tiou_thresholds"]]
    paths = load_config(args.paths)
    cfg["pose_version"] = args.pose_version or cfg.get("pose_version") or paths["versions"]["pose2d"]
    root = Path(paths["raw"]["wayup"])
    pose_dir = Path(paths["interim"]["pose2d"]) / cfg["pose_version"]
    out_dir = Path(paths["results"]) / "track1" / f"{cfg['pose_version']}_{cfg['keypoints']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos, poses, holds, gt = {}, {}, {}, {}
    for rec in wayup.list_recordings(root):
        for route in wayup.ROUTES:
            vid = wayup.video_id(rec, route)
            videos[vid] = rec
            poses[vid] = load_pose2d(pose2d_path(pose_dir, vid))
            holds[vid] = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
            labels = wayup.load_hold_usage(root / rec / f"{route}_holdUsage.csv", vid, poses[vid].num_frames)
            gt[vid] = [le.event for le in labels]

    tuning = cfg["tuning"]
    grid = dict(tuning["grid"])
    model = poses[next(iter(poses))].model
    prefix = next(p for p in tuning["min_score_grid"] if model.startswith(p))
    grid[f"min_score.{prefix}"] = tuning["min_score_grid"][prefix]
    configs = expand_grid(cfg, grid)
    logging.info("%d videos, %d configs per fold", len(videos), len(configs))

    def predict(vid, c):
        return track1_events(poses[vid], holds[vid], c)

    result = lopo_tune(videos, predict, gt, configs, list(grid), wayup.participant_of, thresholds,
                       tuning["objective"])

    preds = result.test_predictions
    write_events([e for vid in videos for e in preds[vid]], out_dir / "events_lopo.csv")
    with (out_dir / "chosen_params.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test_participant", *grid])
        writer.writeheader()
        for p, values in sorted(result.chosen.items()):
            writer.writerow({"test_participant": p, **values})

    rows = []
    for name, limbs in (("all", None), ("hands", HANDS), ("feet", FEET)):
        m = macro_metrics(video_counts(preds[v], gt[v], thresholds, limbs) for v in videos)
        rows.append((f"LOPO-tuned, {name}", m))
    default = {v: track1_events(poses[v], holds[v], cfg) for v in videos}
    rows.append(("default config, all", macro_metrics(video_counts(default[v], gt[v], thresholds) for v in videos)))

    header = "| Setting | " + " | ".join(f"P / R / F1 @{'>0' if t == 0 else f'≥{t}'}" for t in thresholds)
    lines = [f"# Track 1 — {cfg['pose_version']}, {cfg['keypoints']}", "",
             f"Macro average over {len(videos)} videos, LOPO ({len(result.chosen)} folds), "
             f"objective {tuning['objective']}.", "", header + f" | mean tIoU of matches @≥{thresholds[-1]} |",
             "|" + " --- |" * (len(thresholds) + 2)]
    for name, m in rows:
        cells = [f"{m[t]['precision']:.3f} / {m[t]['recall']:.3f} / **{m[t]['f1']:.3f}**" for t in thresholds]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {m[thresholds[-1]]['mean_tiou']:.3f} |")
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
