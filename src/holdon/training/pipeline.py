"""End-to-end Track 2 run on The Way Up: load data, build features, LOPO train / tune, write results.

Called by ``scripts/run_track2.py``; kept here so the script only parses arguments.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from holdon.data import wayup
from holdon.data.external import load_pose2d, pose2d_path
from holdon.data.video import video_info
from holdon.evaluation.grouped import experience_groups, load_experience
from holdon.evaluation.report import breakdown_report, load_ground_truth
from holdon.events import write_events
from holdon.features.build import build_video_table, concat
from holdon.training.train import lopo_train_predict

logger = logging.getLogger(__name__)


def candidate_report(stats: Mapping[str, Mapping[str, int]], n_rows: int, n_pos: int) -> list[str]:
    """Markdown lines: ground-truth contact frames kept / lost by the candidate selection."""
    lines = ["# Track 2 candidate coverage", "",
             f"Feature table: {n_rows} candidate rows, {n_pos} labelled as contact.", "",
             "Ground-truth contact frames (limb on a hold; wall pushes `w` excluded). A frame whose hold is "
             "not a candidate cannot be predicted by Track 2.", "",
             "| Limbs | GT frames | in candidates | lost: no pose | lost: hold not a candidate |",
             "| --- | --- | --- | --- | --- |"]
    groups = dict(stats)
    groups["all"] = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    for name in ("all", "hands", "feet"):
        s = groups[name]
        total = max(s["gt_frames"], 1)
        other = s["gt_frames"] - s["gt_frames_in_candidates"] - s["gt_frames_no_pose"]
        lines.append(f"| {name} | {s['gt_frames']} | {s['gt_frames_in_candidates'] / total:.3f} | "
                     f"{s['gt_frames_no_pose'] / total:.3f} | {other / total:.3f} |")
    return lines


def run_track2(cfg: dict[str, Any], eval_cfg: Mapping[str, Any], paths: Mapping[str, Any]) -> Path:
    """Run Track 2 with ``configs/track2.yaml`` and return the output folder."""
    if cfg.get("model", "svm") != "svm":
        raise NotImplementedError(f"model {cfg['model']!r}: only svm exists in v1")
    cfg["pose_version"] = cfg.get("pose_version") or paths["versions"]["pose2d"]
    thresholds = [float(t) for t in eval_cfg["tiou_thresholds"]]
    root = Path(paths["raw"]["wayup"])
    pose_dir = Path(paths["interim"]["pose2d"]) / cfg["pose_version"]
    out_dir = Path(paths["results"]) / "track2" / f"{cfg['pose_version']}_{cfg['run_name']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    labeled = load_ground_truth(root, lambda rec, route: frames.setdefault(
        wayup.video_id(rec, route), video_info(wayup.video_path(root, rec, route)).num_frames))
    recording_of, tables, fps = {}, {}, None
    for vid in labeled:
        rec, route = vid.rsplit("_", 1)
        recording_of[vid] = rec
        pose = load_pose2d(pose2d_path(pose_dir, vid))
        if pose.num_frames != frames[vid]:
            raise ValueError(f"{vid}: pose has {pose.num_frames} frames, video {frames[vid]}")
        fps = pose.fps
        holds = wayup.load_holds(root / rec / f"{route}_routeAnnotation.json")
        tables[vid] = build_video_table(pose, holds, labeled[vid], cfg)
    full = concat(list(tables.values()))
    logger.info("%d videos, %d candidate rows (%d positive), %d features: %s", len(tables), len(full),
                int(full.label.sum()), len(full.names), ", ".join(full.names))

    result = lopo_train_predict(tables, labeled, frames, recording_of, wayup.participant_of, cfg, thresholds, fps)

    write_events([e for vid in labeled for e in result.test_predictions.get(vid, [])], out_dir / "events_lopo.csv")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    keys = list(cfg["tuning"]["grid"])
    with (out_dir / "chosen_params.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test_participant", *keys, "inner_objective", "train_rows"])
        writer.writeheader()
        for f in sorted(result.folds, key=lambda f: f.test_participant):
            writer.writerow({"test_participant": f.test_participant, **{k: f.params[k] for k in keys},
                             "inner_objective": round(f.inner_objective, 4), "train_rows": f.train_rows})
    (out_dir / "candidates.md").write_text(
        "\n".join(candidate_report(full.stats, len(full), int(full.label.sum()))) + "\n", encoding="utf-8")

    exp = eval_cfg.get("experience")
    groups = (experience_groups(load_experience(root / exp["form"], exp["column"]), exp["min_participants"])
              if exp else None)
    lines = breakdown_report(result.test_predictions, labeled, thresholds,
                             title=f"Track 2 — {cfg['pose_version']}, {cfg['run_name']} (SVM, LOPO)",
                             experience_groups=groups)
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_dir)
    return out_dir
