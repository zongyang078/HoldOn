"""Tests for the temporary Track 1 scoring and LOPO tuning helpers."""

from __future__ import annotations

import pytest

from holdon.data.wayup import participant_of
from holdon.events import ContactEvent
from holdon.models.track1_tuning import (
    expand_grid,
    lopo_tune,
    macro_metrics,
    match_events,
    objective_value,
    tiou,
    video_counts,
)


def ev(start, end, hold="1", limb="left_hand", score=1.0, vid="v"):
    return ContactEvent(vid, limb, hold, start, end, score)


def test_tiou_inclusive_intervals():
    assert tiou(ev(0, 9), ev(0, 9)) == 1.0
    assert tiou(ev(0, 9), ev(5, 14)) == pytest.approx(5 / 15)
    assert tiou(ev(0, 4), ev(5, 9)) == 0.0


def test_match_requires_same_limb_and_hold_and_is_greedy_by_score():
    gt = [ev(0, 9)]
    pred = [ev(0, 9, hold="2"), ev(0, 9, limb="right_hand"), ev(5, 9, score=0.9), ev(0, 9, score=0.5)]
    pairs = match_events(pred, gt, 0.0)
    assert pairs == [(2, 0, 0.5)]  # highest-score eligible prediction wins, gt used once


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


def test_expand_grid_dotted_keys():
    configs = expand_grid({"a": 1, "s": {"w": 3}}, {"a": [1, 2], "s.w": [5]})
    assert configs == [{"a": 1, "s": {"w": 5}}, {"a": 2, "s": {"w": 5}}]


def test_lopo_tune_picks_config_on_training_participants_only():
    videos = {"p1_green": "p1", "p2a_green": "p2a", "p2b_green": "p2b"}
    gt = {v: [ev(0, 9, vid=v)] for v in videos}

    def predict(vid, cfg):  # config "good" is right on p1, "bad" is right on p2a/p2b
        right = (cfg["mode"] == "good") == (vid == "p1_green")
        return [ev(0, 9, vid=vid)] if right else [ev(50, 59, vid=vid)]

    result = lopo_tune(videos, predict, gt, [{"mode": "good"}, {"mode": "bad"}], ["mode"],
                       participant_of, [0.5], "mean_f1")
    # holding out p1, training on p2a+p2b prefers "bad", which then fails on p1
    assert result.chosen == {1: {"mode": "bad"}, 2: {"mode": "good"}}
    assert result.test_predictions["p1_green"][0].start_frame == 50
