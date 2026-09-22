"""Render compact Stage-2C policy and intervention diagnostics."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ACTIONS = ("COMMIT", "REGENERATE", "STRONG_VERIFY", "TOOL_RETRY", "REOBSERVE_REPLAN")
POLICIES = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE", "ATTRIBUTION_DEGRADED", "SHUFFLED_ATTRIBUTION", "ORACLE_ATTRIBUTION")
SOURCES = ("SUCCESS", "G", "V", "X", "E")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def utility_plot(rows: list[dict[str, str]]) -> None:
    subset = [row for row in rows if row["scenario"] == "ID" and row["effectiveness_regime"] == "BASE" and row["cost_regime"] == "BASE"]
    subset.sort(key=lambda row: POLICIES.index(row["policy"]))
    values = [float(row["cost_adjusted_utility"]) for row in subset]
    fig, axis = plt.subplots(figsize=(10.5, 4), constrained_layout=True)
    axis.bar(range(len(subset)), values, color=["#8c8c8c" if row["policy"] != "ATTRIBUTION_AWARE" else "#d73027" for row in subset])
    axis.set(xticks=range(len(subset)), xticklabels=[row["policy"].replace("_", "\n") for row in subset], ylabel="Cost-adjusted utility", title="ID BASE: utility by policy")
    axis.tick_params(axis="x", labelsize=8)
    fig.savefig(OUT / "utility_by_policy.png", dpi=160)
    plt.close(fig)


def confusion_plot(rows: list[dict[str, str]]) -> None:
    selected = ["UNCERTAINTY_ONLY", "ATTRIBUTION_AWARE"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, policy in zip(axes, selected):
        matrix = np.zeros((len(SOURCES), len(ACTIONS)))
        for row in rows:
            if row["policy"] == policy:
                matrix[SOURCES.index(row["true_source"]), ACTIONS.index(row["action"])] = float(row["rate_within_cause"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set(title=policy.replace("_", " "), xticks=range(len(ACTIONS)), xticklabels=[name.replace("_", "\n") for name in ACTIONS], yticks=range(len(SOURCES)), yticklabels=SOURCES)
        axis.tick_params(axis="x", labelsize=8)
        for row in range(len(SOURCES)):
            for column in range(len(ACTIONS)):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, label="Action rate within true cause")
    fig.savefig(OUT / "intervention_confusion_matrix.png", dpi=160)
    plt.close(fig)


def cost_regime_plot(rows: list[dict[str, str]]) -> None:
    costs = ("BASE", "CHEAP_STRONG", "HIGH_INTERVENTION")
    effects = ("BASE", "V_STRONG")
    matrix = np.zeros((len(effects), len(costs)), dtype=int)
    labels = []
    candidates = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE")
    for row_index, effect in enumerate(effects):
        for column_index, cost in enumerate(costs):
            subset = [row for row in rows if row["scenario"] == "ID" and row["effectiveness_regime"] == effect and row["cost_regime"] == cost and row["policy"] in candidates]
            best = max(subset, key=lambda row: float(row["cost_adjusted_utility"]))["policy"]
            matrix[row_index, column_index] = candidates.index(best)
            labels.append((row_index, column_index, best))
    fig, axis = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    axis.imshow(matrix, cmap="tab10", vmin=0, vmax=len(candidates)-1, aspect="auto")
    axis.set(title="Best simple/full policy across cost and effect regimes", xticks=range(len(costs)), xticklabels=costs, yticks=range(len(effects)), yticklabels=effects)
    for row_index, column_index, label in labels:
        axis.text(column_index, row_index, label.replace("_", "\n"), ha="center", va="center", fontsize=8)
    fig.savefig(OUT / "cost_regime_map.png", dpi=160)
    plt.close(fig)


def recovery_plot(rows: list[dict[str, str]]) -> None:
    subset = [row for row in rows if row["scenario"] == "ID" and row["policy"] == "ATTRIBUTION_AWARE"]
    labels = [f"{row['effectiveness_regime']}\n{row['cost_regime']}" for row in subset]
    values = [float(row["oracle_recovery_ratio"]) if row["oracle_recovery_ratio"] else np.nan for row in subset]
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.bar(range(len(values)), values, color="#d73027")
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    axis.set(xticks=range(len(values)), xticklabels=labels, ylabel="Oracle utility gap recovered", title="Attribution-aware oracle recovery")
    axis.tick_params(axis="x", labelsize=8)
    fig.savefig(OUT / "oracle_recovery.png", dpi=160)
    plt.close(fig)


def ood_plot(rows: list[dict[str, str]]) -> None:
    selected = ("UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE", "ORACLE_ATTRIBUTION")
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    positions = np.arange(len(selected))
    for offset, scenario in [(-.18, "ID"), (.18, "OOD_PREVALENCE")]:
        subset = [next(row for row in rows if row["scenario"] == scenario and row["effectiveness_regime"] == "BASE" and row["cost_regime"] == "BASE" and row["policy"] == policy) for policy in selected]
        axis.bar(positions + offset, [float(row["cost_adjusted_utility"]) for row in subset], .34, label=scenario)
    axis.set(xticks=positions, xticklabels=[name.replace("_", "\n") for name in selected], ylabel="Cost-adjusted utility", title="ID versus OOD prevalence shift")
    axis.tick_params(axis="x", labelsize=8)
    axis.legend()
    figure.savefig(OUT / "ood_results.png", dpi=160)
    plt.close(figure)


def main() -> None:
    rows = read(OUT / "policy_results.csv")
    confusion = read(OUT / "intervention_confusion.csv")
    utility_plot(rows)
    confusion_plot(confusion)
    cost_regime_plot(rows)
    recovery_plot(rows)
    ood_plot(rows)


if __name__ == "__main__":
    main()
