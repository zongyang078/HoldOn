"""Tests for keypoint speed, body scale, hold ranking and candidate selection."""

from __future__ import annotations

import numpy as np

from holdon.features.overlap2d import candidate_mask, count_within, distance_rank, mean_point
from holdon.features.stillness import body_scale, keypoint_speed, normalized_speed


def test_keypoint_speed_central_and_missing():
    xy = np.array([[0, 0], [3, 4], [6, 8], [np.nan, np.nan], [0, 0]], dtype=float)
    v = keypoint_speed(xy)
    assert v[0] == 5 and v[1] == 5  # one-sided at the start, central inside
    assert np.isnan(v[2]) and np.isnan(v[4])


def test_body_scale_and_normalized_speed():
    bbox = np.tile([0, 0, 10, 200], (5, 1)).astype(float)
    bbox[2] = np.nan
    scale = body_scale(bbox, window=3)
    assert scale[0] == 200 and np.isnan(scale[2])
    xy = np.column_stack([np.arange(5) * 20.0, np.zeros(5)])
    assert normalized_speed(xy, scale)[1] == 0.1


def test_distance_rank_tiebreak_and_nan_last():
    dist = np.array([[0.0, 0.0, 5.0, np.nan]])
    tiebreak = np.array([[3.0, 1.0, 0.0, 0.0]])
    assert distance_rank(dist, tiebreak).tolist() == [[1, 0, 2, 3]]


def test_candidate_mask_radius_and_k():
    dist = np.array([[0.0, 10.0, 50.0, 100.0], [np.nan] * 4])
    rank = distance_rank(dist)
    assert candidate_mask(dist, rank, radius=80, max_candidates=2).tolist() == [
        [True, True, False, False], [False] * 4]
    assert candidate_mask(dist, rank, radius=80, max_candidates=5)[0].tolist() == [True, True, True, False]
    assert count_within(dist, 80).tolist() == [3, 0]


def test_mean_point_ignores_missing():
    a = np.array([[0.0, 0.0], [np.nan, np.nan]])
    b = np.array([[2.0, 2.0], [np.nan, np.nan]])
    c = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    m = mean_point([a, b, c])
    assert m[0].tolist() == [1.0, 1.0] and np.isnan(m[1]).all()
