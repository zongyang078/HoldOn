"""Track 2 reference classifier built on scikit-learn SVM.

Pipeline: median imputation of unknown features -> StandardScaler -> Nystroem RBF kernel
approximation -> LinearSVC with balanced class weights. The kernel approximation keeps training
linear in the number of rows; if the training set is larger than ``max_train_rows`` it is
subsampled with the class ratio kept (stratified).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_svm(cfg: Mapping[str, Any], seed: int = 0) -> Pipeline:
    """Unfitted pipeline from ``configs/track2.yaml`` -> ``svm``."""
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Nystroem(kernel="rbf", gamma=cfg.get("gamma"), n_components=int(cfg["nystroem_components"]),
                 random_state=seed),
        LinearSVC(C=float(cfg.get("C", 1.0)), class_weight="balanced", max_iter=int(cfg.get("max_iter", 5000)),
                  random_state=seed),
    )


def stratified_subsample(y: np.ndarray, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    """Row indices: all rows if ``len(y) <= max_rows``, else a sample keeping the class ratio."""
    y = np.asarray(y, dtype=bool)
    if max_rows is None or len(y) <= max_rows:
        return np.arange(len(y))
    picked = []
    for cls in (False, True):
        idx = np.flatnonzero(y == cls)
        n = int(round(max_rows * len(idx) / len(y)))
        picked.append(rng.choice(idx, size=min(max(n, 1), len(idx)), replace=False))
    return np.sort(np.concatenate(picked))


def fit_svm(X: np.ndarray, y: np.ndarray, cfg: Mapping[str, Any], seed: int = 0) -> Pipeline:
    """Fit the pipeline on (a stratified subsample of) ``X``, ``y``."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        raise ValueError("Training labels contain a single class")
    rows = stratified_subsample(y, cfg.get("max_train_rows"), np.random.default_rng(seed))
    model = make_svm(cfg, seed)
    model.fit(np.asarray(X)[rows], y[rows])
    return model


def decision_scores(model: Pipeline, X: np.ndarray, batch: int = 200_000) -> np.ndarray:
    """Signed distance to the SVM margin (higher = more likely a contact), computed in batches."""
    X = np.asarray(X)
    if len(X) == 0:
        return np.zeros(0)
    return np.concatenate([model.decision_function(X[i : i + batch]) for i in range(0, len(X), batch)])
