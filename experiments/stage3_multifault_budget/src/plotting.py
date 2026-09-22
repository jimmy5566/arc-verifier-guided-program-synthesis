"""Render Stage-3 budget, OOD, responsibility, and oracle-gap figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
COMPONENTS = ("G", "V", "X", "E")
ACTIONS = ("REGENERATE", "STRONG_VERIFY", "TOOL_RETRY", "REOBSERVE_REPLAN")
POLICIES = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE", "SHUFFLED_ATTRIBUTION", "ORACLE_MULTI_LABEL")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def confusion(rows: list[dict[str, str]]) -> None:
    selected = ("UNCERTAINTY_ONLY", "ATTRIBUTION_AWARE")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, policy in zip(axes, selected):
        matrix = np.zeros((4, 4))
        for row in rows:
            if row["policy"] == policy:
                matrix[COMPONENTS.index(row["true_component"]), ACTIONS.index(row["action"])] = float(row["rate_within_component"])
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set(title=policy.replace("_", " "), xticks=range(4), xticklabels=[a.replace("_", "\n") for a in ACTIONS], yticks=range(4), yticklabels=COMPONENTS)
        for r in range(4):
            for c in range(4): axis.text(c, r, f"{matrix[r,c]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, label="Action use within active component")
    figure.savefig(OUT / "intervention_confusion_matrix.png", dpi=160)
    plt.close(figure)


def utility_by_policy(rows: list[dict[str, str]]) -> None:
    subset = [r for r in rows if r["scenario"] == "ID" and r["effectiveness_regime"] == "BASE" and r["cost_regime"] == "MEDIUM" and r["budget"] == "1"]
    subset.sort(key=lambda r: POLICIES.index(r["policy"]))
    fig, ax = plt.subplots(figsize=(10.5, 4), constrained_layout=True)
    ax.bar(range(len(subset)), [float(r["cost_adjusted_utility"]) for r in subset], color=["#d73027" if r["policy"] == "ATTRIBUTION_AWARE" else "#8c8c8c" for r in subset])
    ax.set(xticks=range(len(subset)), xticklabels=[r["policy"].replace("_", "\n") for r in subset], ylabel="Cost-adjusted utility", title="ID BASE, budget 1")
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(OUT / "utility_by_policy.png", dpi=160); plt.close(fig)


def regime_map(rows: list[dict[str, str]]) -> None:
    effects = ("BASE", "V_STRONG"); costs = ("LOW", "MEDIUM", "HIGH"); budgets = (1, 2)
    candidates = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, budget in zip(axes, budgets):
        labels = []; values = np.zeros((2,3), dtype=int)
        for i, effect in enumerate(effects):
            for j, cost in enumerate(costs):
                subset = [r for r in rows if r["scenario"] == "ID" and r["effectiveness_regime"] == effect and r["cost_regime"] == cost and r["budget"] == str(budget) and r["policy"] in candidates]
                winner = max(subset, key=lambda r: float(r["cost_adjusted_utility"]))["policy"]
                values[i,j] = candidates.index(winner); labels.append((i,j,winner))
        ax.imshow(values, cmap="tab10", vmin=0, vmax=len(candidates)-1, aspect="auto")
        ax.set(title=f"Best policy, budget {budget}", xticks=range(3), xticklabels=costs, yticks=range(2), yticklabels=effects)
        for i,j,label in labels: ax.text(j,i,label.replace("_","\n"),ha="center",va="center",fontsize=8)
    fig.savefig(OUT / "cost_regime_map.png", dpi=160); plt.close(fig)


def oracle_recovery(rows: list[dict[str, str]]) -> None:
    subset = [r for r in rows if r["scenario"] == "ID" and r["policy"] == "ATTRIBUTION_AWARE"]
    fig, ax = plt.subplots(figsize=(11,4), constrained_layout=True)
    values = [float(r["oracle_recovery_ratio"]) if r["oracle_recovery_ratio"] else np.nan for r in subset]
    labels = [f"b{r['budget']}\n{r['effectiveness_regime']}\n{r['cost_regime']}" for r in subset]
    ax.bar(range(len(values)),values,color="#d73027"); ax.axhline(1,color="black",linestyle="--",linewidth=1)
    ax.set(xticks=range(len(values)),xticklabels=labels,ylabel="Oracle gap recovered",title="Attribution-aware recovery of Oracle gain")
    ax.tick_params(axis="x",labelsize=7)
    fig.savefig(OUT / "oracle_recovery.png",dpi=160);plt.close(fig)


def ood(rows: list[dict[str,str]]) -> None:
    selected = ("UNCERTAINTY_ONLY","FAILURE_RISK_ONLY","STATE_ONLY","ATTRIBUTION_AWARE","ORACLE_MULTI_LABEL")
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True); pos=np.arange(len(selected))
    for offset, scenario in [(-.18,"ID"),(.18,"OOD_DUAL_SHIFT")]:
        subset=[next(r for r in rows if r["scenario"]==scenario and r["effectiveness_regime"]=="BASE" and r["cost_regime"]=="MEDIUM" and r["budget"]=="2" and r["policy"]==policy) for policy in selected]
        ax.bar(pos+offset,[float(r["cost_adjusted_utility"]) for r in subset],.34,label=scenario)
    ax.set(xticks=pos,xticklabels=[p.replace("_","\n") for p in selected],ylabel="Cost-adjusted utility",title="Budget 2: ID versus OOD dual-fault shift");ax.tick_params(axis="x",labelsize=8);ax.legend()
    fig.savefig(OUT / "ood_results.png",dpi=160);plt.close(fig)


def main() -> None:
    results=read(OUT/"policy_results.csv"); conf=read(OUT/"component_action_confusion.csv")
    confusion(conf); utility_by_policy(results); regime_map(results); oracle_recovery(results); ood(results)


if __name__ == "__main__": main()
