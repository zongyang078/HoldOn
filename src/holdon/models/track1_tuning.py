"""Leave-one-participant-out threshold tuning for Track 1.

Scoring (matching, P / R / F1, macro average, tuning objective) comes from the shared
``holdon.evaluation`` module (docs/evaluation_protocol.md); this module only expands the parameter
grid and picks the best config on the training participants of each fold.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from holdon.data.splits import lopo_folds
from holdon.evaluation.metrics import macro_metrics, objective_value, video_counts
from holdon.events import ContactEvent


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
