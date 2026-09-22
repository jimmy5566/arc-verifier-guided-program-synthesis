"""Plot the frozen Stage-2B deployable-policy utility regime map."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPAIRS = [0.50, 0.70, 0.90, 1.00]
COSTS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30]
POLICY_CODES = {
    "STATE_NAIVE": 0,
    "PREDICTED_ATTRIBUTION": 1,
    "ALWAYS_STRONG": 2,
    "NEVER_STRONG": 3,
}
POLICY_LABELS = ["State-naive", "Predicted attribution", "Always strong", "Never strong"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    regime = rows(OUT / "regime_map.csv")
    recovery = rows(OUT / "oracle_recovery.csv")
    by_cell = {(float(row["repair_probability"]), float(row["verifier_cost"])): row for row in regime}
    by_recovery = {(float(row["repair_probability"]), float(row["verifier_cost"])): row for row in recovery}
    winner = np.zeros((len(REPAIRS), len(COSTS)))
    advantage = np.zeros_like(winner, dtype=float)
    for row_index, repair in enumerate(REPAIRS):
        for column_index, cost in enumerate(COSTS):
            label = by_cell[(repair, cost)]["best_deployable_policy"]
            # Ties retain the first named deployable policy only for rendering.
            winner[row_index, column_index] = POLICY_CODES[label.split("+")[0]]
            advantage[row_index, column_index] = float(by_recovery[(repair, cost)]["predicted_minus_naive"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    image = axes[0].imshow(winner, cmap="tab10", vmin=0, vmax=3, aspect="auto")
    axes[0].set(title="Best deployable policy by fixed utility", xlabel="Strong-verifier cost", ylabel="Repair probability", xticks=range(len(COSTS)), xticklabels=COSTS, yticks=range(len(REPAIRS)), yticklabels=REPAIRS)
    for row_index in range(len(REPAIRS)):
        for column_index in range(len(COSTS)):
            axes[0].text(column_index, row_index, POLICY_LABELS[int(winner[row_index, column_index])], ha="center", va="center", fontsize=7)

    scale = max(abs(advantage.min()), abs(advantage.max()))
    image = axes[1].imshow(advantage, cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
    axes[1].set(title="Predicted attribution utility minus State-naive", xlabel="Strong-verifier cost", ylabel="Repair probability", xticks=range(len(COSTS)), xticklabels=COSTS, yticks=range(len(REPAIRS)), yticklabels=REPAIRS)
    for row_index in range(len(REPAIRS)):
        for column_index in range(len(COSTS)):
            axes[1].text(column_index, row_index, f"{advantage[row_index, column_index]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], label="Utility difference")
    fig.savefig(OUT / "utility_regime_map.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
