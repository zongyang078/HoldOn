"""Tests for keypoint-to-hold-box distances."""

from __future__ import annotations

import numpy as np

from holdon.data.wayup import Hold
from holdon.features.overlap2d import hold_boxes, min_distance, point_box_distance


def test_point_box_distance_inside_outside_missing():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    xy = np.array([[5, 5], [13, 14], [10, 0], [np.nan, np.nan]])
    d = point_box_distance(xy, boxes)[:, 0]
    assert d[0] == 0 and d[1] == 5 and d[2] == 0 and np.isnan(d[3])


def test_hold_boxes_and_min_distance():
    boxes = hold_boxes([Hold("0", 0, 0, 10, 10, False), Hold("1", 20, 0, 30, 10, True)])
    assert boxes.shape == (2, 4)
    a = np.array([[5, 5], [np.nan, np.nan]])
    b = np.array([[25, 5], [np.nan, np.nan]])
    d = min_distance([a, b], boxes)
    assert d[0].tolist() == [0, 0]  # each box touched by one of the two points
    assert np.isnan(d[1]).all()
