"""Telemetry-only attribution models for preregistered Stage-1."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CLASSES = np.array(["success", "generator", "selector", "executor", "environment"])


def fit_models(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    majority = np.full(len(CLASSES), 1 / len(CLASSES), dtype=float)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    majority = counts / counts.sum()
    # Older installed scikit-learn versions infer multiclass behavior and do
    # not accept the newer explicit ``multi_class`` keyword.
    logistic = LogisticRegression(max_iter=800, random_state=seed)
    forest = RandomForestClassifier(n_estimators=160, min_samples_leaf=8, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    logistic.fit(x, y); forest.fit(x, y)
    return {"majority": majority, "logistic": logistic, "random_forest": forest}


def probabilities(model: object, x: np.ndarray) -> np.ndarray:
    if isinstance(model, np.ndarray):
        return np.tile(model, (len(x), 1))
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), len(CLASSES)))
    aligned[:, model.classes_] = raw
    return aligned
