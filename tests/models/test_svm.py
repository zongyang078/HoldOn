"""Tests for the Track 2 SVM pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from holdon.models.svm import decision_scores, fit_svm, stratified_subsample

CFG = {"nystroem_components": 20, "gamma": None, "C": 1.0, "max_iter": 2000, "max_train_rows": 200}


def test_stratified_subsample_keeps_ratio():
    y = np.array([True] * 100 + [False] * 900)
    idx = stratified_subsample(y, 100, np.random.default_rng(0))
    assert len(idx) == 100 and y[idx].sum() == 10
    assert len(stratified_subsample(y, None, np.random.default_rng(0))) == 1000


def test_fit_svm_separates_and_handles_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] > 0.5
    X[::17, 1] = np.nan  # imputed by the pipeline
    model = fit_svm(X, y, CFG)
    s = decision_scores(model, X)
    assert s.shape == (400,) and ((s > 0) == y).mean() > 0.9
    assert decision_scores(model, X[:0]).shape == (0,)
    with pytest.raises(ValueError):
        fit_svm(X, np.zeros(400, bool), CFG)
