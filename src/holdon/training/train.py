"""Train Track 2 classifiers fold by fold under the LOPO protocol.

Outer loop: leave-one-participant-out (10 folds on The Way Up, ``p2a`` + ``p2b`` together). Inside a
fold only the training participants are used to choose the decoding parameters: they are split into
``inner_folds`` participant groups, the classifier is trained on all but one group and scores the
remaining one, giving out-of-fold scores for every training row; ``threshold``, ``min_duration_s``
and ``merge_gap_frames`` are tuned on those scores. The classifier is then refit on all training
participants and applied to the held-out participant with the chosen parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from holdon.data.splits import lopo_folds
from holdon.evaluation.metrics import GroundTruth
from holdon.events import ContactEvent
from holdon.features.build import FeatureTable, concat
from holdon.models.svm import decision_scores, fit_svm
from holdon.training.tune import best_holds, decode_all, tune_decoding

logger = logging.getLogger(__name__)


def participant_groups(participants: Sequence[int], n_groups: int) -> list[list[int]]:
    """Split participants into ``n_groups`` groups round-robin in sorted order (deterministic)."""
    ordered = sorted(set(participants))
    n = min(n_groups, len(ordered))
    if n < 2:
        raise ValueError(f"Need at least 2 participants for inner folds, got {ordered}")
    return [ordered[i::n] for i in range(n)]


def out_of_fold_scores(
    table: FeatureTable, participant_of_row: np.ndarray, n_groups: int, svm_cfg: Mapping[str, Any], seed: int,
) -> np.ndarray:
    """(N,) classifier scores where each row is scored by a model trained without its participant group."""
    scores = np.full(len(table), np.nan)
    for g, group in enumerate(participant_groups(participant_of_row.tolist(), n_groups)):
        held = np.isin(participant_of_row, group)
        model = fit_svm(table.X[~held], table.label[~held], svm_cfg, seed + g)
        scores[held] = decision_scores(model, table.X[held])
    return scores


@dataclass
class FoldResult:
    """What one outer fold chose and how it did on its training participants."""

    test_participant: int
    params: dict[str, Any]
    inner_objective: float
    train_rows: int


@dataclass
class Track2Result:
    """Held-out predictions of every video plus the per-fold choices."""

    folds: list[FoldResult] = field(default_factory=list)
    test_predictions: dict[str, list[ContactEvent]] = field(default_factory=dict)
    test_scores: dict[str, np.ndarray] = field(default_factory=dict)  # video -> (rows,) scores of its table


def lopo_train_predict(
    tables: Mapping[str, FeatureTable],
    gt_by_video: Mapping[str, Sequence[GroundTruth]],
    num_frames: Mapping[str, int],
    recording_of: Mapping[str, str],
    participant_of: Callable[[str], int],
    cfg: Mapping[str, Any],
    thresholds: Sequence[float],
    fps: float,
) -> Track2Result:
    """Run the full Track 2 LOPO protocol.

    Args:
        tables: video id -> its feature table (``holdon.features.build.build_video_table``).
        gt_by_video: video id -> ground truth.
        num_frames: video id -> number of frames.
        recording_of: video id -> recording (``p2b_orange`` -> ``p2b``).
        participant_of: recording -> participant number.
        cfg: ``configs/track2.yaml`` (uses ``svm``, ``inner_folds``, ``decoding``, ``tuning``, ``seed``).
        thresholds: tIoU thresholds of the tuning objective.
        fps: video frame rate (for ``min_duration_s``).
    """
    seed = int(cfg.get("seed", 0))
    result = Track2Result()
    for fold in lopo_folds(set(recording_of.values()), participant_of):
        train_v = [v for v in tables if recording_of[v] in fold.train]
        test_v = [v for v in tables if recording_of[v] in fold.test]
        train = concat([tables[v] for v in train_v])
        row_participant = np.array([participant_of(recording_of[v]) for v in train.video])

        oof = out_of_fold_scores(train, row_participant, int(cfg["inner_folds"]), cfg["svm"], seed)
        best = best_holds(train, oof, {v: num_frames[v] for v in train_v})
        params, value = tune_decoding(best, gt_by_video, cfg["decoding"], cfg["tuning"]["grid"], thresholds,
                                      cfg["tuning"]["objective"], fps)

        model = fit_svm(train.X, train.label, cfg["svm"], seed)
        test = concat([tables[v] for v in test_v])
        scores = decision_scores(model, test.X)
        preds = decode_all(best_holds(test, scores, {v: num_frames[v] for v in test_v}), params, fps)
        result.test_predictions.update(preds)
        for v in test_v:
            result.test_scores[v] = scores[test.video == v]
        result.folds.append(FoldResult(fold.test_participant, params, value, len(train)))
        logger.info("fold p%d: %s (inner %s %.3f, %d train rows)", fold.test_participant, params,
                    cfg["tuning"]["objective"], value, len(train))
    return result
