"""Small, deterministic Matplotlib figures for the Stage-0 pilot."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def failure_decomposition_plot(rows: Iterable[dict], output: Path) -> None:
    counts = Counter(str(row["failure_source"]) for row in rows)
    labels = ["success", "selection_failure", "generator_limited_failure"]
    values = [counts[label] for label in labels]
    colors = ["#4c78a8", "#f58518", "#e45756"]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Frozen ARC30 tasks")
    axis.set_title("Frozen ARC30: candidate-generation vs selection attribution")
    axis.set_ylim(0, max(values) + 3)
    axis.tick_params(axis="x", labelrotation=18)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value), ha="center")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def contamination_plot(rows: Iterable[dict], output: Path) -> None:
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    figure, axis = plt.subplots(figsize=(7.3, 4.5))
    palette = {"STATIC": "#666666", "NAIVE": "#e45756", "ORACLE_SKIP": "#54a24b", "ORACLE_CLEAN": "#4c78a8"}
    for method, method_rows in sorted(by_method.items()):
        method_rows.sort(key=lambda row: float(row["rho"]))
        x = [float(row["rho"]) for row in method_rows]
        y = [float(row["absolute_error_mean"]) for row in method_rows]
        low = [max(0.0, abs(float(row["bias_lower_95"]))) for row in method_rows]
        high = [abs(float(row["bias_upper_95"])) for row in method_rows]
        axis.plot(x, y, marker="o", label=method.replace("_", "-"), color=palette.get(method))
        if method == "NAIVE":
            # The asymmetric band is displayed around the mean estimate error
            # through the corresponding reliability-bias interval.
            axis.fill_between(x, low, high, color=palette[method], alpha=0.15)
    axis.set_xlabel("Synthetic one-sided external failure probability (rho)")
    axis.set_ylabel("Absolute reliability-estimation error vs clean selector target")
    axis.set_title("Sensitivity to outcome contamination (Monte Carlo, n=30)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
