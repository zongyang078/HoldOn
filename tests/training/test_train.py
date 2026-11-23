"""Tests for Track 2 decoding, tuning and the LOPO training loop (synthetic videos, CPU, seconds)."""

from __future__ import annotations

import numpy as np

from holdon.data.wayup import participant_of
from holdon.features.build import build_video_table
from holdon.models.decoding import best_hold_per_frame, decode_best_holds
from holdon.training.train import lopo_train_predict, participant_groups
from holdon.training.tune import best_holds, tune_decoding
from tests.conftest import make_climb


def test_best_hold_per_frame_and_decoding():
    frames = np.array([0, 0, 1, 2, 3, 4])
    holds = np.array(["a", "b", "b", "b", "a", "b"], dtype=object)
    scores = np.array([0.2, 0.9, 0.8, -1.0, 0.7, 0.6])
    best, s, ids = best_hold_per_frame(frames, holds, scores, 6)
    assert ids == ["a", "b"] and best.tolist() == [1, 1, 1, 0, 1, -1]
    assert s[5] == -np.inf
    ev = decode_best_holds(best, s, ids, video_id="v", limb="left_hand", threshold=0.0, min_length=1, max_gap=0)
    assert [(e.hold_id, e.start_frame, e.end_frame) for e in ev] == [("a", 3, 3), ("b", 0, 1), ("b", 4, 4)]
    assert abs(ev[1].score - 0.85) < 1e-12  # mean frame score of the event


def test_participant_groups_round_robin():
    assert participant_groups([5, 1, 3, 3, 2], 2) == [[1, 3], [2, 5]]


def test_tune_decoding_uses_given_videos(track2_cfg):
    pose, holds, labeled = make_climb("p1_green")
    table = build_video_table(pose, holds, labeled, track2_cfg)
    scores = np.where(table.label, 1.0, -1.0)
    best = best_holds(table, scores, {"p1_green": 60})
    params, value = tune_decoding(best, {"p1_green": labeled}, track2_cfg["decoding"],
                                  {"threshold": [2.0, 0.0]}, [0.0, 0.5], "mean_f1", 25.0)
    assert params["threshold"] == 0.0 and value == 1.0


def test_lopo_train_predict_end_to_end(track2_cfg):
    videos = {"p1_green": "p1", "p2a_green": "p2a", "p2b_green": "p2b", "p3_green": "p3"}
    tables, gt = {}, {}
    for i, vid in enumerate(videos):
        pose, holds, labeled = make_climb(vid, seed=i)
        tables[vid], gt[vid] = build_video_table(pose, holds, labeled, track2_cfg), labeled
    result = lopo_train_predict(tables, gt, {v: 60 for v in videos}, videos, participant_of, track2_cfg,
                                [0.0, 0.5], 25.0)
    assert sorted(f.test_participant for f in result.folds) == [1, 2, 3]
    assert set(result.test_predictions) == set(videos)
    for vid, events in result.test_predictions.items():
        hits = [e for e in events if e.limb == "left_hand" and e.hold_id == "0"]
        assert hits and hits[0].start_frame <= 12 and hits[0].end_frame >= 37
        assert len(result.test_scores[vid]) == len(tables[vid])
