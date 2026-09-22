"""Metrics and table helpers for Stage-1."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from reliability_update import ece


def attribution_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    selector = (labels == 2).astype(int)
    per_class = f1_score(labels, predicted, labels=range(5), average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predicted)), "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)), "f1_success": float(per_class[0]), "f1_generator": float(per_class[1]), "f1_selector": float(per_class[2]), "f1_executor": float(per_class[3]), "f1_environment": float(per_class[4]), "selector_f1": float(f1_score(selector, predicted == 2, zero_division=0)), "selector_auroc": float(roc_auc_score(selector, probabilities[:, 2])), "selector_brier": float(np.mean((probabilities[:, 2] - selector) ** 2)), "selector_ece": ece(probabilities[:, 2], selector), "confusion": confusion_matrix(labels, predicted, labels=range(5)).tolist()}
