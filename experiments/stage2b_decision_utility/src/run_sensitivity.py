"""Stage-2B CPU-only decision-utility sensitivity using frozen Stage-2A logic."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STAGE2A_SRC = ROOT.parents[0] / "stage2a_state_conditional_attribution" / "src"
sys.path.insert(0, str(STAGE2A_SRC))
from simulate import STATES, episodes, proba  # noqa: E402

REPAIR_PROBABILITIES = (0.50, 0.70, 0.90, 1.00)
VERIFIER_COSTS = (0.00, 0.02, 0.05, 0.10, 0.20, 0.30)
THRESHOLD = 0.70
POLICIES = ("STATE_NAIVE", "PREDICTED_ATTRIBUTION", "ORACLE_ATTRIBUTION", "ALWAYS_STRONG", "NEVER_STRONG")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_policy(
    source: np.ndarray,
    state: np.ndarray,
    prediction: np.ndarray,
    policy: str,
    repair_probability: float,
    verifier_cost: float,
    repair_draw: np.ndarray,
) -> dict[str, float | int | str]:
    """Run one online policy. Hidden source is never read by predicted updates."""
    alpha = np.ones(4)
    beta = np.ones(4)
    success: list[bool] = []
    calls: list[bool] = []
    wrong_commits: list[bool] = []
    for index, (hidden_source, context, probabilities) in enumerate(zip(source, state, prediction)):
        if policy == "ALWAYS_STRONG":
            call = True
        elif policy == "NEVER_STRONG":
            call = False
        else:
            reliability = alpha[context] / (alpha[context] + beta[context])
            call = reliability < THRESHOLD
        episode_success = bool(repair_draw[index] < repair_probability) if call and hidden_source == 2 else hidden_source == 0
        success.append(episode_success)
        calls.append(call)
        wrong_commits.append((not call) and hidden_source == 2)

        if policy == "STATE_NAIVE":
            positive, negative = ((1.0, 0.0) if hidden_source == 0 else (0.0, 1.0))
        elif policy == "ORACLE_ATTRIBUTION":
            if hidden_source == 1:  # G: no selector evidence, oracle only.
                continue
            positive, negative = ((1.0, 0.0) if hidden_source != 2 else (0.0, 1.0))
        elif policy == "PREDICTED_ATTRIBUTION":
            # Telemetry-only probabilistic attribution; no hidden source access.
            positive = float(probabilities[0] + probabilities[3] + probabilities[4])
            negative = float(probabilities[2])
        else:
            continue
        alpha[context] += positive
        beta[context] += negative

    call_rate = float(np.mean(calls))
    success_rate = float(np.mean(success))
    total_cost = verifier_cost * call_rate
    return {
        "policy": policy,
        "repair_probability": repair_probability,
        "verifier_cost": verifier_cost,
        "threshold": THRESHOLD,
        "task_success": success_rate,
        "wrong_commit_rate": float(np.mean(wrong_commits)),
        "strong_call_rate": call_rate,
        "total_verification_cost": total_cost,
        "cost_adjusted_utility": success_rate - total_cost,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_x, _, calibration_y = episodes(8000, 2201)
    # The validation split is deliberately materialized but never used to tune a threshold.
    validation_x, validation_state, validation_y = episodes(2000, 2202)
    test_x, test_state, test_y = episodes(12000, 2203)
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=2201,
        class_weight="balanced_subsample",
    ).fit(calibration_x, calibration_y)
    prediction = proba(model, test_x)
    repair_draw = np.random.default_rng(2205).random(len(test_y))

    rows: list[dict] = []
    for repair_probability in REPAIR_PROBABILITIES:
        for verifier_cost in VERIFIER_COSTS:
            for policy in POLICIES:
                rows.append(evaluate_policy(test_y, test_state, prediction, policy, repair_probability, verifier_cost, repair_draw))
    write_csv(OUT / "policy_sensitivity.csv", rows)

    # Predicted attribution's recovery is meaningful only where the oracle has
    # a positive advantage over State-Naive under the same frozen utility.
    recovery: list[dict] = []
    regimes: list[dict] = []
    for repair_probability in REPAIR_PROBABILITIES:
        for verifier_cost in VERIFIER_COSTS:
            subset = [row for row in rows if row["repair_probability"] == repair_probability and row["verifier_cost"] == verifier_cost]
            by_policy = {row["policy"]: row for row in subset}
            naive = by_policy["STATE_NAIVE"]["cost_adjusted_utility"]
            predicted = by_policy["PREDICTED_ATTRIBUTION"]["cost_adjusted_utility"]
            oracle = by_policy["ORACLE_ATTRIBUTION"]["cost_adjusted_utility"]
            denom = oracle - naive
            recovery.append({
                "repair_probability": repair_probability,
                "verifier_cost": verifier_cost,
                "state_naive_utility": naive,
                "predicted_utility": predicted,
                "oracle_utility": oracle,
                "predicted_minus_naive": predicted - naive,
                "oracle_minus_naive": denom,
                "oracle_benefit_recovered_by_predicted": (predicted - naive) / denom if denom > 0 else None,
            })
            deployable = {name: by_policy[name]["cost_adjusted_utility"] for name in ("STATE_NAIVE", "PREDICTED_ATTRIBUTION", "ALWAYS_STRONG", "NEVER_STRONG")}
            best_value = max(deployable.values())
            winners = "+".join(sorted(name for name, value in deployable.items() if np.isclose(value, best_value, atol=1e-12)))
            regimes.append({"repair_probability": repair_probability, "verifier_cost": verifier_cost, "best_deployable_policy": winners, "best_utility": best_value, "predicted_strictly_best": deployable["PREDICTED_ATTRIBUTION"] > max(value for name, value in deployable.items() if name != "PREDICTED_ATTRIBUTION")})
    write_csv(OUT / "oracle_recovery.csv", recovery)
    write_csv(OUT / "regime_map.csv", regimes)
    metadata = {
        "status": "PASS",
        "sandbox": "Stage-2A imported unchanged",
        "calibration_episodes": len(calibration_y),
        "validation_episodes_unused": len(validation_y),
        "test_episodes": len(test_y),
        "repair_probabilities": REPAIR_PROBABILITIES,
        "verifier_costs": VERIFIER_COSTS,
        "threshold_not_tuned": THRESHOLD,
        "predicted_strictly_best_cell_count": sum(row["predicted_strictly_best"] for row in regimes),
        "outputs": ["policy_sensitivity.csv", "oracle_recovery.csv", "regime_map.csv"],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
