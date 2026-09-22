"""Frozen60-specific CPU-only figures; historical Frozen30 code is untouched."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def failure_decomposition(rows: Iterable[dict], output: Path) -> None:
    counts = Counter(str(row["failure_source"]) for row in rows)
    labels = ["success", "selection_failure", "generator_limited_failure"]
    values = [counts[label] for label in labels]
    figure, axis = plt.subplots(figsize=(7.1, 4.2))
    bars = axis.bar(labels, values, color=["#4c78a8", "#f58518", "#e45756"])
    axis.set_title("Frozen60 failure attribution (B-SUPPORT, 32 views)")
    axis.set_ylabel("Tasks"); axis.set_ylim(0, max(values) + 5); axis.tick_params(axis="x", labelrotation=16)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.4, str(value), ha="center")
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def top2_hierarchy(rows: Iterable[dict], output: Path) -> None:
    rows = list(rows)
    top1 = sum(bool(row["top1_correct"]) for row in rows)
    rescued = sum(bool(row["top2_correct"]) and not bool(row["top1_correct"]) for row in rows)
    below = sum(bool(row["any_of_k_correct"]) and not bool(row["top2_correct"]) for row in rows)
    figure, axis = plt.subplots(figsize=(7.1, 4.2))
    label = ["Correct candidate exists (Any-of-K)"]
    axis.bar(label, [top1], label="Top-1 correct", color="#4c78a8")
    axis.bar(label, [rescued], bottom=[top1], label="Top-2 rescue", color="#54a24b")
    axis.bar(label, [below], bottom=[top1 + rescued], label="Correct candidate below Top-2", color="#f58518")
    axis.set_ylabel("Tasks"); axis.set_ylim(0, top1 + rescued + below + 4)
    axis.set_title("Frozen60 hierarchy: generation -> ranking -> commitment")
    axis.legend(loc="upper right")
    for y, value in ((top1 / 2, top1), (top1 + rescued / 2, rescued), (top1 + rescued + below / 2, below)):
        axis.text(0, y, str(value), ha="center", va="center", color="white", fontweight="bold")
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def contamination(rows: Iterable[dict], output: Path) -> None:
    methods: dict[str, list[dict]] = {}
    for row in rows:
        methods.setdefault(str(row["method"]), []).append(row)
    colors = {"STATIC": "#666666", "NAIVE": "#e45756", "ORACLE_SKIP": "#54a24b", "ORACLE_CLEAN": "#4c78a8"}
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    for method, method_rows in sorted(methods.items()):
        method_rows.sort(key=lambda row: float(row["rho"]))
        axis.plot([float(row["rho"]) for row in method_rows], [float(row["absolute_error_mean"]) for row in method_rows], marker="o", label=method.replace("_", "-"), color=colors[method])
    axis.set_xlabel("Synthetic external failure probability (rho)")
    axis.set_ylabel("Absolute error vs clean selector correctness")
    axis.set_title("Frozen60 one-sided contamination sensitivity")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def cohort_comparison(rows: Iterable[dict], output: Path) -> None:
    rows = list(rows); names = [str(row["cohort"]) for row in rows]
    metrics = [("any_of_k_rate", "Any-of-K"), ("top1_rate", "Top-1"), ("top2_rate", "Top-2")]
    x = list(range(len(names))); width = 0.23
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for index, (metric, label) in enumerate(metrics):
        values = []
        missing = []
        for position, row in enumerate(rows):
            raw = row.get(metric)
            if raw in (None, ""):
                values.append(0.0); missing.append(position)
            else:
                values.append(float(raw))
        bars = axis.bar([value + (index - 1) * width for value in x], values, width, label=label)
        for position in missing:
            bar = bars[position]
            axis.text(bar.get_x() + bar.get_width() / 2, 0.025, "N/A", ha="center", va="bottom", rotation=90, fontsize=8)
    axis.set_xticks(x, names); axis.set_ylim(0, 1.0); axis.set_ylabel("Task-level rate")
    axis.set_title("Task-disjoint Frozen30 vs Frozen60 comparison")
    axis.legend(); axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)
