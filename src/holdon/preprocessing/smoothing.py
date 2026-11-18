"""Temporal smoothing of keypoint trajectories (Gaussian and median filters).

All functions work on ``(T, D)`` float arrays (usually ``D = 2`` for pixel x, y) where NaN marks a
frame without a usable keypoint. NaN-aware filters average only the valid samples in each window,
and frames that are NaN after gap interpolation stay NaN: smoothing never invents a position across
a long gap.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def mask_low_confidence(xy: np.ndarray, scores: np.ndarray, min_score: float) -> np.ndarray:
    """Return a copy of ``xy`` (T, D) with rows whose score is below ``min_score`` set to NaN."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[np.asarray(scores) < min_score] = np.nan
    return out


def interpolate_gaps(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of at most ``max_gap`` frames that have valid frames on both sides."""
    out = np.array(xy, dtype=np.float64, copy=True)
    if max_gap <= 0:
        return out
    valid = ~np.isnan(out).any(axis=1)
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            out[a + 1 : b] = (1 - w) * out[a] + w * out[b]
    return out


def _windows(xy: np.ndarray, window: int) -> np.ndarray:
    """Centred sliding windows (T, D, window), padded with NaN at both ends."""
    half = window // 2
    padded = np.pad(xy, ((half, window - 1 - half), (0, 0)), constant_values=np.nan)
    return sliding_window_view(padded, window, axis=0)


def median_filter(xy: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred moving median over ``window`` frames; NaN input frames stay NaN."""
    xy = np.asarray(xy, dtype=np.float64)
    if window <= 1 or len(xy) == 0:
        return xy.copy()
    with np.errstate(all="ignore"), _quiet_nan_warnings():
        out = np.nanmedian(_windows(xy, window), axis=-1)
    out[np.isnan(xy)] = np.nan
    return out


def gaussian_filter(xy: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware Gaussian smoothing, kernel truncated to ``window`` frames, sigma = window / 4."""
    xy = np.asarray(xy, dtype=np.float64)
    if window <= 1 or len(xy) == 0:
        return xy.copy()
    sigma = window / 4.0
    offsets = np.arange(window) - window // 2
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    win = _windows(xy, window)
    weights = np.where(np.isnan(win), 0.0, kernel)
    with np.errstate(all="ignore"):
        out = np.nansum(win * kernel, axis=-1) / weights.sum(axis=-1)
    out[np.isnan(xy)] = np.nan
    return out


def smooth_track(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    min_score: float,
    max_gap: int,
    method: str,
    window: int,
) -> np.ndarray:
    """Full keypoint clean-up: drop low-confidence frames, fill short gaps, then smooth.

    Args:
        xy: (T, 2) keypoint positions, NaN where missing.
        scores: (T,) keypoint confidences (scale depends on the pose model).
        min_score: frames below this confidence are treated as missing.
        max_gap: longest run of missing frames that is linearly interpolated.
        method: ``"median"``, ``"gaussian"`` or ``"none"``.
        window: filter length in frames.
    """
    out = interpolate_gaps(mask_low_confidence(xy, scores, min_score), max_gap)
    if method == "median":
        return median_filter(out, window)
    if method == "gaussian":
        return gaussian_filter(out, window)
    if method == "none":
        return out
    raise ValueError(f"Unknown smoothing method {method!r}")


class _quiet_nan_warnings:
    """Silence 'All-NaN slice' RuntimeWarnings from nanmedian on empty windows."""

    def __enter__(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(*exc)
