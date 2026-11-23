"""Tests for tIoU and greedy event matching (moved from tests/models/test_track1_tuning.py)."""

from __future__ import annotations

import pytest

from holdon.evaluation.matching import match_events, tiou
from holdon.events import ContactEvent


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


def test_each_prediction_takes_best_unused_gt():
    gt = [ev(0, 9), ev(8, 20)]
    pairs = match_events([ev(8, 20, score=0.9), ev(0, 10, score=0.8)], gt, 0.5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]
