"""Leave-one-participant-out threshold tuning and scoring for Track 1.

TEMPORARY scoring: ``match_events`` / ``video_counts`` / ``macro_metrics`` are a minimal
implementation of docs/evaluation_protocol.md so Track 1 can be tuned and reported in Phase 2.
They will be replaced by the shared ``holdon.evaluation`` module (owned by the learning line);
do not build other tracks on them.

Protocol implemented here: a prediction matches only a ground-truth event with the same limb and
hold; tIoU on inclusive frame intervals; greedy matching by descending prediction score, each event
used once; TP / FP / FN per video; Precision, Recall, F1 per video, macro-averaged over videos.
Threshold 0.0 means tIoU > 0, any other t means tIoU >= t.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from holdon.data.splits import lopo_folds
from holdon.events import ContactEvent

HANDS = ("left_hand", "right_hand")
FEET = ("left_foot", "right_foot")


def tiou(a: ContactEvent, b: ContactEvent) -> float:
    """Temporal IoU of two inclusive frame intervals."""
    inter = min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame) + 1
    if inter <= 0:
        return 0.0
    union = max(a.end_frame, b.end_frame) - min(a.start_frame, b.start_frame) + 1
    return inter / union


def _passes(value: float, threshold: float) -> bool:
    return value > 0 if threshold == 0 else value >= threshold


def match_events(
    pred: Sequence[ContactEvent], gt: Sequence[ContactEvent], threshold: float
) -> list[tuple[int, int, float]]:
    """Greedy matching; returns ``(pred index, gt index, tIoU)`` for each matched pair."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i].score)
    used: set[int] = set()
    pairs = []
    for i in order:
        p = pred[i]
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt):
            if j in used or g.limb != p.limb or g.hold_id != p.hold_id:
                continue
            value = tiou(p, g)
            if value > best_iou:
                best, best_iou = j, value
        if best >= 0 and _passes(best_iou, threshold):
            used.add(best)
            pairs.append((i, best, best_iou))
    return pairs


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
    """Counts per tIoU threshold for one video, optionally restricted to some limbs."""
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


def expand_grid(base: Mapping[str, Any], grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """All configs obtained by setting each dotted key in ``grid`` (e.g. ``smoothing.window``)."""
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(dict(base))
        for key, value in zip(keys, values):
            node = cfg
            *parents, leaf = key.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = value
        configs.append(cfg)
    return configs


@dataclass
class LopoResult:
    """Outcome of LOPO tuning: chosen config per fold and held-out predictions per video."""

    chosen: dict[int, dict[str, Any]]  # participant -> chosen values of the grid keys
    test_predictions: dict[str, list[ContactEvent]]  # video id -> predictions of its test fold


def lopo_tune(
    videos: Mapping[str, str],
    predict: Callable[[str, Mapping[str, Any]], list[ContactEvent]],
    ground_truth: Mapping[str, Sequence[ContactEvent]],
    configs: Sequence[Mapping[str, Any]],
    grid_keys: Sequence[str],
    participant_of: Callable[[str], int],
    thresholds: Sequence[float],
    objective: str,
) -> LopoResult:
    """Pick the best config on the training participants of each fold and apply it to the test one.

    Args:
        videos: video id -> recording name (``p2b_orange`` -> ``p2b``).
        predict: ``predict(video_id, config)`` -> events for that video.
        ground_truth: video id -> ground-truth events.
        configs: candidate configs (see ``expand_grid``).
        grid_keys: dotted keys that vary between configs, recorded in ``chosen``.
    """
    counts = {  # (config index, video) -> counts; every config is run once per video
        (c, vid): video_counts(predict(vid, cfg), ground_truth[vid], thresholds)
        for c, cfg in enumerate(configs) for vid in videos
    }
    result = LopoResult({}, {})
    for fold in lopo_folds(set(videos.values()), participant_of):
        train = [v for v, rec in videos.items() if rec in fold.train]
        test = [v for v, rec in videos.items() if rec in fold.test]
        scores = [objective_value(macro_metrics(counts[(c, v)] for v in train), objective)
                  for c in range(len(configs))]
        best = int(np.argmax(scores))
        result.chosen[fold.test_participant] = {k: _get(configs[best], k) for k in grid_keys}
        for vid in test:
            result.test_predictions[vid] = predict(vid, configs[best])
    return result


def _get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node
