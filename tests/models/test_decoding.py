"""Tests for turning per-frame masks into contact events."""

from __future__ import annotations

import numpy as np

from holdon.models.decoding import decode_events, mask_to_intervals, merge_intervals, min_frames


def test_mask_to_intervals_edges():
    mask = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)
    assert mask_to_intervals(mask) == [(0, 1), (4, 4), (6, 6)]
    assert mask_to_intervals(np.zeros(3, bool)) == []


def test_merge_and_min_frames():
    assert merge_intervals([(0, 2), (5, 6), (10, 12)], max_gap=2) == [(0, 6), (10, 12)]
    assert min_frames(0.5, 25) == 13 and min_frames(0.4, 25) == 10


def test_decode_events_min_length_and_score():
    mask = np.zeros(40, bool)
    mask[0:5] = True  # too short
    mask[10:20] = True
    mask[21:30] = True  # merged with the previous run across a 1-frame gap
    scores = np.full(40, 0.5)
    events = decode_events(mask, video_id="v", limb="left_hand", hold_id="3",
                           min_length=13, max_gap=1, frame_scores=scores)
    assert [(e.start_frame, e.end_frame) for e in events] == [(10, 29)]
    assert events[0].score == 0.5 and events[0].hold_id == "3"
