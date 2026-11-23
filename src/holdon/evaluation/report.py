"""Ground-truth loading and Markdown metric tables shared by scripts/evaluate.py and the tracks.

The table format matches the Track 1 ``metrics.md``: one row per setting, ``P / R / **F1**`` per
tIoU threshold, then the mean tIoU of matched pairs at the strictest threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

from holdon.data import wayup
from holdon.data.wayup import LabeledEvent
from holdon.evaluation import grouped
from holdon.evaluation.metrics import FEET, HANDS
from holdon.events import ContactEvent

Metrics = Mapping[float, Mapping[str, float]]


def load_ground_truth(
    root: str | Path, num_frames: Callable[[str, str], int] | None = None,
) -> dict[str, list[LabeledEvent]]:
    """Labeled events of every The Way Up video (22), keyed by video id.

    Args:
        root: dataset root (``configs/paths.yaml`` -> ``raw.wayup``).
        num_frames: ``num_frames(recording, route)`` used to clip end frames past the video, e.g.
            from ``holdon.data.video.video_info``; None = no clipping.
    """
    root = Path(root)
    out = {}
    for rec in wayup.list_recordings(root):
        for route in wayup.ROUTES:
            vid = wayup.video_id(rec, route)
            n = num_frames(rec, route) if num_frames is not None else None
            out[vid] = wayup.load_hold_usage(root / rec / f"{route}_holdUsage.csv", vid, n)
    return out


def threshold_label(t: float) -> str:
    """``0.0`` -> ``>0``, ``0.3`` -> ``≥0.3``."""
    return ">0" if t == 0 else f"≥{t}"


def metrics_table(rows: Sequence[tuple[str, Metrics]], thresholds: Sequence[float],
                  first_column: str = "Setting") -> list[str]:
    """Markdown table lines: one row per ``(name, metrics)``."""
    lines = [f"| {first_column} | " + " | ".join(f"P / R / F1 @{threshold_label(t)}" for t in thresholds)
             + f" | mean tIoU of matches @{threshold_label(thresholds[-1])} |",
             "|" + " --- |" * (len(thresholds) + 2)]
    for name, m in rows:
        cells = [f"{m[t]['precision']:.3f} / {m[t]['recall']:.3f} / **{m[t]['f1']:.3f}**" for t in thresholds]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {m[thresholds[-1]]['mean_tiou']:.3f} |")
    return lines


def breakdown_report(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    labeled_by_video: Mapping[str, Sequence[LabeledEvent]],
    thresholds: Sequence[float],
    *,
    title: str,
    experience_groups: Mapping[int, str] | None = None,
) -> list[str]:
    """Full Markdown report: all / hands / feet, per route, occlusion frame recall, experience."""
    thresholds = [float(t) for t in thresholds]
    n_videos = len(labeled_by_video)
    n_pred = sum(len(pred_by_video.get(v, [])) for v in labeled_by_video)
    n_gt = sum(len(v) for v in labeled_by_video.values())
    n_wall = sum(le.event.hold_id == wayup.WALL_HOLD_ID for v in labeled_by_video.values() for le in v)
    lines = [f"# {title}", "",
             f"Macro average over {n_videos} videos; {n_pred} predicted and {n_gt} ground-truth events "
             f"({n_wall} wall pushes `w`, never predicted, count as FN). Protocol: docs/evaluation_protocol.md.",
             "", "## All limbs, hands, feet", ""]
    limb_rows = grouped.by_limb_type(pred_by_video, labeled_by_video, thresholds)
    lines += metrics_table(list(limb_rows.items()), thresholds)

    lines += ["", "## Route (macro over the route's videos)", ""]
    route_rows = grouped.by_route(pred_by_video, labeled_by_video, thresholds)
    lines += metrics_table([(f"{r} ({sum(grouped.route_of(v) == r for v in labeled_by_video)} videos)", m)
                            for r, m in route_rows.items()], thresholds, "Route")

    lines += ["", "## Occluded vs visible contact frames (frame recall, pooled over all events)", "",
              "| Limbs | occluded frames | recall occluded | visible frames | recall visible |",
              "| --- | --- | --- | --- | --- |"]
    for name, limbs in (("all", None), ("hands", HANDS), ("feet", FEET)):
        fr = grouped.occlusion_frame_recall(pred_by_video, labeled_by_video, limbs)
        o, v = fr["occluded"], fr["visible"]
        lines.append(f"| {name} | {o.frames} | {o.recall:.3f} | {v.frames} | {v.recall:.3f} |")

    if experience_groups is not None:
        lines += ["", "## Climber experience (group averages only; macro over the group's videos)", ""]
        exp = grouped.by_experience(pred_by_video, labeled_by_video, experience_groups, wayup.participant_of,
                                    thresholds)
        lines += metrics_table([(f"{g} ({p} participants, {nv} videos)", m) for g, (p, nv, m) in exp.items()],
                               thresholds, "Climbing for")
    return lines
