"""Tests for per-video counts, macro averaging and the ``evaluate`` entry point."""

from __future__ import annotations

import pytest

from holdon.data.wayup import LabeledEvent
from holdon.evaluation.metrics import HANDS, evaluate, macro_metrics, objective_value, video_counts
from holdon.events import ContactEvent


def ev(start, end, hold="1", limb="left_hand", score=1.0, vid="v"):
    return ContactEvent(vid, limb, hold, start, end, score)


def test_threshold_semantics_and_counts():
    gt, pred = [ev(0, 9)], [ev(7, 16)]  # tIoU = 3 / 17
    counts = video_counts(pred, gt, [0.0, 0.3])
    assert (counts[0.0].tp, counts[0.0].fp, counts[0.0].fn) == (1, 0, 0)
    assert (counts[0.3].tp, counts[0.3].fp, counts[0.3].fn) == (0, 1, 1)
    hands_only = video_counts([ev(0, 9, limb="left_foot")], gt, [0.0], limbs=("left_hand",))
    assert hands_only[0.0].fp == 0 and hands_only[0.0].fn == 1


def test_macro_average_over_videos():
    perfect = video_counts([ev(0, 9)], [ev(0, 9)], [0.5])
    empty = video_counts([], [ev(0, 9)], [0.5])
    m = macro_metrics([perfect, empty])[0.5]
    assert m["f1"] == 0.5 and m["recall"] == 0.5 and m["mean_tiou"] == 1.0
    assert objective_value({0.5: m}, "f1@0.5") == 0.5


def test_evaluate_macro_over_gt_videos_and_labeled_events():
    gt = {"a": [LabeledEvent(ev(0, 9, vid="a"), ((0, 4),))], "b": [ev(0, 9, vid="b")]}
    pred = {"a": [ev(0, 9, vid="a")]}  # video b has no predictions: P = R = F1 = 0
    table = evaluate(pred, gt)
    assert set(table) == {0.0, 0.3, 0.5}
    assert table[0.5] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "mean_tiou": 1.0}


def test_evaluate_limb_filter_and_unknown_video():
    gt = {"a": [ev(0, 9, vid="a"), ev(0, 9, limb="left_foot", vid="a")]}
    pred = {"a": [ev(0, 9, vid="a")]}
    assert evaluate(pred, gt, [0.5], limbs=HANDS)[0.5]["f1"] == 1.0
    assert evaluate(pred, gt, [0.5])[0.5]["recall"] == 0.5
    with pytest.raises(ValueError):
        evaluate({"zzz": []}, gt)
