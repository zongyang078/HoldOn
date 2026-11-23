"""Tests for the Track 1 LOPO tuning helpers (scoring tests live in tests/evaluation/)."""

from __future__ import annotations

from holdon.data.wayup import participant_of
from holdon.events import ContactEvent
from holdon.models.track1_tuning import expand_grid, lopo_tune


def ev(start, end, hold="1", limb="left_hand", score=1.0, vid="v"):
    return ContactEvent(vid, limb, hold, start, end, score)


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
