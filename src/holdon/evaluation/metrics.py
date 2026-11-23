"""Precision / recall / F1 per video and their macro average (docs/evaluation_protocol.md).

Per video: ``P = TP / (TP + FP)``, ``R = TP / (TP + FN)``, ``F1 = 2PR / (P + R)``, each 0 when its
denominator is 0. Macro average over videos (every video weighs the same, none is skipped). Mean
tIoU: average over all matched pairs of all videos at the given threshold.

``Counts`` … ``objective_value`` were moved unchanged from ``holdon.models.track1_tuning``;
``evaluate`` is the entry point of docs/interfaces.md section 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Union

import numpy as np

from holdon.data.wayup import LabeledEvent
from holdon.evaluation.matching import match_events
from holdon.events import ContactEvent

HANDS: tuple[str, ...] = ("left_hand", "right_hand")
FEET: tuple[str, ...] = ("left_foot", "right_foot")
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.3, 0.5)

GroundTruth = Union[ContactEvent, LabeledEvent]


@dataclass(frozen=True)
class Counts:
    """Matching outcome of one video at one tIoU threshold."""

    tp: int
    fp: int
    fn: int
    tiou_sum: float

    def prf(self) -> tuple[float, float, float]:
        """Precision, recall, F1 (0 when undefined)."""
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f


def video_counts(
    pred: Sequence[ContactEvent], gt: Sequence[ContactEvent], thresholds: Iterable[float],
    limbs: Sequence[str] | None = None,
) -> dict[float, Counts]:
    """Counts per tIoU threshold for one video, optionally restricted to some limbs.

    The limb filter applies to **both** predictions and ground truth.
    """
    if limbs is not None:
        pred = [e for e in pred if e.limb in limbs]
        gt = [e for e in gt if e.limb in limbs]
    out = {}
    for t in thresholds:
        pairs = match_events(pred, gt, t)
        out[t] = Counts(len(pairs), len(pred) - len(pairs), len(gt) - len(pairs), sum(v for *_, v in pairs))
    return out


def macro_metrics(per_video: Iterable[Mapping[float, Counts]]) -> dict[float, dict[str, float]]:
    """Macro average over videos: mean P, R, F1 per threshold, plus mean tIoU over all matches."""
    per_video = list(per_video)
    out = {}
    for t in per_video[0]:
        prfs = np.array([v[t].prf() for v in per_video])
        tp = sum(v[t].tp for v in per_video)
        out[t] = {
            "precision": float(prfs[:, 0].mean()),
            "recall": float(prfs[:, 1].mean()),
            "f1": float(prfs[:, 2].mean()),
            "mean_tiou": sum(v[t].tiou_sum for v in per_video) / tp if tp else 0.0,
        }
    return out


def objective_value(metrics: Mapping[float, Mapping[str, float]], objective: str) -> float:
    """``mean_f1`` (mean F1 over all thresholds) or ``f1@<t>`` (F1 at one threshold)."""
    if objective == "mean_f1":
        return float(np.mean([m["f1"] for m in metrics.values()]))
    if objective.startswith("f1@"):
        return metrics[float(objective[3:])]["f1"]
    raise ValueError(f"Unknown objective {objective!r}")


def as_events(gt: Iterable[GroundTruth]) -> list[ContactEvent]:
    """Ground truth as plain ContactEvents (``LabeledEvent`` -> ``.event``)."""
    return [g.event if isinstance(g, LabeledEvent) else g for g in gt]


def evaluate(
    pred_by_video: Mapping[str, Sequence[ContactEvent]],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    limbs: Sequence[str] | None = None,
) -> dict[float, dict[str, float]]:
    """Score predictions against ground truth, macro-averaged over the videos of ``gt_by_video``.

    Args:
        pred_by_video: video id -> predicted events. A video without an entry has no predictions.
        gt_by_video: video id -> ground truth (``ContactEvent`` or ``LabeledEvent``); its keys are
            the videos that are averaged (none is skipped).
        thresholds: tIoU thresholds (``0.0`` = tIoU > 0).
        limbs: restrict predictions and ground truth to these limbs (e.g. ``HANDS``).

    Returns:
        ``{t: {"precision", "recall", "f1", "mean_tiou"}}``.
    """
    if not gt_by_video:
        raise ValueError("gt_by_video is empty")
    unknown = sorted(set(pred_by_video) - set(gt_by_video))
    if unknown:
        raise ValueError(f"Predictions for videos without ground truth: {unknown}")
    thresholds = [float(t) for t in thresholds]
    return macro_metrics(
        video_counts(list(pred_by_video.get(v, [])), as_events(gt), thresholds, limbs)
        for v, gt in gt_by_video.items()
    )
