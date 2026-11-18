"""Tests for NaN-aware keypoint smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.preprocessing.smoothing import (
    gaussian_filter,
    interpolate_gaps,
    mask_low_confidence,
    median_filter,
    smooth_track,
)


def _xy(values):
    v = np.asarray(values, dtype=float)
    return np.column_stack([v, v * 2])


def test_mask_low_confidence():
    out = mask_low_confidence(_xy([1, 2, 3]), np.array([0.9, 0.1, 0.9]), 0.5)
    assert np.isnan(out[1]).all() and out[2].tolist() == [3, 6]


def test_interpolate_short_gaps_only():
    xy = _xy([0, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7])
    out = interpolate_gaps(xy, max_gap=2)
    assert out[:4, 0].tolist() == [0, 1, 2, 3]  # gap of 2 filled
    assert np.isnan(out[4:7]).all()  # gap of 3 kept


def test_interpolate_does_not_extrapolate_edges():
    out = interpolate_gaps(_xy([np.nan, 1, 2, np.nan]), max_gap=5)
    assert np.isnan(out[0]).all() and np.isnan(out[3]).all()


def test_median_removes_spike_and_keeps_nan():
    xy = _xy([1, 1, 50, 1, 1, np.nan, 1])
    out = median_filter(xy, 3)
    assert out[2, 0] == 1 and np.isnan(out[5]).all()


def test_gaussian_preserves_constant_and_length():
    xy = _xy([4.0] * 10)
    out = gaussian_filter(xy, 5)
    assert out.shape == xy.shape and np.allclose(out, xy)


def test_smooth_track_pipeline_and_bad_method():
    xy = _xy([1, 1, 1, 1, 1])
    scores = np.array([1, 1, 0, 1, 1.0])
    out = smooth_track(xy, scores, min_score=0.5, max_gap=1, method="median", window=3)
    assert np.allclose(out[:, 0], 1)  # low-score frame masked, then interpolated
    with pytest.raises(ValueError):
        smooth_track(xy, scores, min_score=0.5, max_gap=1, method="kalman", window=3)
