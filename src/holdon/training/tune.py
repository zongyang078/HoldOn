"""Tune decoding parameters on training folds only (threshold, minimum duration, gap merging).

Classifier scores are first reduced to the best candidate hold per limb and frame
(``holdon.models.decoding.best_hold_per_frame``); every grid point is then decoded and scored with
``holdon.evaluation.metrics.evaluate`` on the training videos, and the best one is returned.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from holdon.events import LIMBS, ContactEvent
from holdon.evaluation.metrics import GroundTruth, evaluate, objective_value
from holdon.features.build import FeatureTable
from holdon.models.decoding import best_hold_per_frame, decode_best_holds, min_frames
from holdon.models.track1_tuning import expand_grid

BestHolds = dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]]


def best_holds(table: FeatureTable, scores: np.ndarray, num_frames: Mapping[str, int]) -> BestHolds:
    """Video -> limb -> ``best_hold_per_frame`` output, for every video in ``num_frames``."""
    out: BestHolds = {}
    for vid, n in num_frames.items():
        in_vid = table.video == vid
        out[vid] = {}
        for li, limb in enumerate(LIMBS):
            rows = in_vid & (table.limb == li)
            out[vid][limb] = best_hold_per_frame(table.frame[rows], table.hold[rows], scores[rows], n)
    return out


def decode_all(best: BestHolds, params: Mapping[str, Any], fps: float) -> dict[str, list[ContactEvent]]:
    """Events of every video for one setting of ``threshold``, ``min_duration_s``, ``merge_gap_frames``."""
    min_length = min_frames(float(params["min_duration_s"]), fps)
    out = {}
    for vid, limbs in best.items():
        events: list[ContactEvent] = []
        for limb, (b, s, ids) in limbs.items():
            events += decode_best_holds(b, s, ids, video_id=vid, limb=limb, threshold=float(params["threshold"]),
                                        min_length=min_length, max_gap=int(params["merge_gap_frames"]))
        out[vid] = sorted(events, key=lambda e: (e.limb, e.start_frame, e.hold_id))
    return out


def tune_decoding(
    best: BestHolds,
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    base: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    thresholds: Sequence[float],
    objective: str,
    fps: float,
) -> tuple[dict[str, Any], float]:
    """Best decoding setting on ``best``'s videos: ``(params, objective value)``.

    ``base`` holds the default ``threshold`` / ``min_duration_s`` / ``merge_gap_frames``; ``grid``
    the values to try for any of them. Ties keep the first setting in grid order.
    """
    gt = {v: gt_by_video[v] for v in best}
    best_value, best_params = -np.inf, dict(base)
    for params in expand_grid(base, grid):
        value = objective_value(evaluate(decode_all(best, params, fps), gt, thresholds), objective)
        if value > best_value:
            best_value, best_params = value, params
    return dict(best_params), float(best_value)
