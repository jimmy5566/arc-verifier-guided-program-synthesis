"""Create compact CPU-only figures from frozen Stage-2A CSV outputs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STATES = ["A_high", "B_medium", "C_low", "D_volatile"]
METHODS = ["STATE_NAIVE", "ORACLE_STATE", "PREDICTED_STATE", "PREDICTED_DEGRADED"]
LABELS = ["State-naive", "Oracle", "Predicted", "Degraded"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = read_rows(OUT / "state_calibration_and_decisions.csv")
    by_key = {(row["method"], row["state"]): row for row in rows}
    positions = np.arange(len(STATES))
    width = 0.19

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    true = [float(by_key[("STATE_NAIVE", state)]["true_reliability"]) for state in STATES]
    axes[0].plot(positions, true, "ko--", label="True reliability")
    for index, (method, label) in enumerate(zip(METHODS, LABELS)):
        estimates = [float(by_key[(method, state)]["final_estimate"]) for state in STATES]
        axes[0].bar(positions + (index - 1.5) * width, estimates, width, label=label, alpha=.85)
    axes[0].set(xticks=positions, xticklabels=[state.split("_")[0] for state in STATES], ylim=(0, 1), ylabel="Reliability", title="State-conditional reliability estimates")
    axes[0].legend(fontsize=8)

    for index, (method, label) in enumerate(zip(METHODS, LABELS)):
        success = [float(by_key[(method, state)]["task_success_rate"]) for state in STATES]
        cost = [float(by_key[(method, state)]["strong_verifier_cost"]) for state in STATES]
        axes[1].bar(positions + (index - 1.5) * width, success, width, label=f"{label}: task success")
        axes[1].plot(positions + (index - 1.5) * width, cost, "x", color="black", ms=5)
    axes[1].set(xticks=positions, xticklabels=[state.split("_")[0] for state in STATES], ylim=(0, 1), ylabel="Rate / cost", title="Decision result (bars) and verifier cost (x)")
    axes[1].legend(fontsize=7)
    fig.savefig(OUT / "state_calibration_and_decisions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
