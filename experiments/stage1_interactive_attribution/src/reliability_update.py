"""Delayed transparent Beta reliability updates, independent of the simulator."""
from __future__ import annotations

import numpy as np


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any(): result += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(result)


def run_updates(labels: np.ndarray, probabilities: np.ndarray, delay: int, method: str, threshold: float = .70) -> dict:
    """Update selector reliability only when delayed feedback is available.

    labels use 0=success,1=generator,2=selector,3=executor,4=environment.
    The target selector correctness excludes generator episodes.
    """
    clean = (labels != 2)[labels != 1].mean()
    alpha = beta = 1.0; scheduled: dict[int, list[tuple[float, float]]] = {}; errors = []; updates = 0
    for tick, source in enumerate(labels):
        for positive, negative in scheduled.pop(tick, []):
            alpha += positive; beta += negative; updates += 1
        errors.append(abs(alpha / (alpha + beta) - clean))
        if method == "STATIC": continue
        if method == "NAIVE":
            positive, negative = (1.0, 0.0) if source == 0 else (0.0, 1.0)
        elif method == "ORACLE":
            if source == 1: continue
            positive, negative = (1.0, 0.0) if source != 2 else (0.0, 1.0)
        else:
            p = probabilities[tick]
            if method == "GATED" and float(p.max()) < threshold: continue
            # Fractional responsibility: P(V) is selector-negative evidence;
            # P(success,X,E) is selector-correct evidence; P(G) abstains.
            positive, negative = float(p[0] + p[3] + p[4]), float(p[2])
        # Delay zero means feedback is available immediately after this
        # decision and therefore informs the next decision.  Scheduling it at
        # the already-passed current tick would silently discard every online
        # update until the tail flush.
        if delay == 0:
            alpha += positive; beta += negative; updates += 1
        else:
            scheduled.setdefault(tick + delay, []).append((positive, negative))
    # Tail feedback is delivered for final posterior, but online error only
    # measures information actually available while decisions were occurring.
    for pending in scheduled.values():
        for positive, negative in pending: alpha += positive; beta += negative; updates += 1
    estimate = alpha / (alpha + beta)
    selector_fail = (labels == 2).astype(float)
    return {"clean_target": float(clean), "final_estimate": float(estimate), "final_abs_error": float(abs(estimate-clean)), "mean_online_abs_error": float(np.mean(errors)), "updates": updates, "selector_brier": float(np.mean((probabilities[:, 2] - selector_fail) ** 2)), "selector_ece": ece(probabilities[:, 2], selector_fail)}
