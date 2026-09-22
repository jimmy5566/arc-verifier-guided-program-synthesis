"""CPU-only Stage-2C responsibility attribution and multi-intervention study."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STAGE2A_SRC = ROOT.parents[0] / "stage2a_state_conditional_attribution" / "src"
sys.path.insert(0, str(STAGE2A_SRC))
from simulate import FEATURES, STATES, episodes, proba  # noqa: E402

SOURCES = ("SUCCESS", "G", "V", "X", "E")
ACTIONS = ("COMMIT", "REGENERATE", "STRONG_VERIFY", "TOOL_RETRY", "REOBSERVE_REPLAN")
POLICIES = (
    "COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY",
    "STATE_ONLY", "ATTRIBUTION_AWARE", "ATTRIBUTION_DEGRADED", "SHUFFLED_ATTRIBUTION",
    "ORACLE_ATTRIBUTION",
)
EFFECTS = {
    "BASE": np.array([
        [.99, .03, .03, .03, .03],
        [.96, .82, .12, .10, .12],
        [.98, .18, .84, .08, .10],
        [.97, .12, .10, .78, .40],
        [.97, .10, .10, .38, .82],
    ]),
    "V_STRONG": np.array([
        [.99, .03, .03, .03, .03],
        [.96, .82, .12, .10, .12],
        [.98, .18, .97, .08, .10],
        [.97, .12, .10, .78, .40],
        [.97, .10, .10, .38, .82],
    ]),
}
COSTS = {
    "BASE": np.array([.00, .05, .15, .08, .10]),
    "CHEAP_STRONG": np.array([.00, .05, .04, .08, .10]),
    "HIGH_INTERVENTION": np.array([.00, .12, .24, .18, .22]),
}
OOD_STATE_P = np.array([.15, .20, .35, .30])
OOD_SOURCE_P = np.array([
    [.45, .10, .13, .16, .16], [.15, .20, .28, .22, .15],
    [.12, .16, .45, .10, .17], [.20, .06, .23, .30, .21],
])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def expected_actions(probabilities: np.ndarray, effects: np.ndarray, costs: np.ndarray) -> np.ndarray:
    return (probabilities @ effects.T - costs).argmax(axis=1)


def uncertainty_actions(x: np.ndarray) -> np.ndarray:
    # Stage-2A columns: margin=5, entropy=6, disagreement=7.
    score = ((1 - x[:, 5]) + (x[:, 6] / 1.2) + x[:, 7]) / 3
    return np.where(score >= .60, 2, 0)


def sample_ood(n: int = 12000, seed: int = 2210) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift only prevalence by resampling a large ID conditional telemetry pool."""
    pool_x, pool_state, pool_source = episodes(100000, seed)
    rng = np.random.default_rng(seed + 1)
    target_state = rng.choice(4, n, p=OOD_STATE_P)
    target_source = np.array([rng.choice(5, p=OOD_SOURCE_P[state]) for state in target_state])
    available = {(state, source): np.flatnonzero((pool_state == state) & (pool_source == source)) for state in range(4) for source in range(5)}
    selected = np.array([rng.choice(available[(state, source)]) for state, source in zip(target_state, target_source)])
    return pool_x[selected], target_state, target_source


def run_policy(
    policy: str,
    x: np.ndarray,
    source: np.ndarray,
    p_full: np.ndarray,
    p_state: np.ndarray,
    p_failure: np.ndarray,
    p_degraded: np.ndarray,
    p_shuffled: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    draw: np.ndarray,
) -> tuple[dict, np.ndarray]:
    if policy == "COMMIT_ONLY":
        action = np.zeros(len(source), dtype=int)
    elif policy == "ALWAYS_STRONG_VERIFY":
        action = np.full(len(source), 2, dtype=int)
    elif policy == "UNCERTAINTY_ONLY":
        action = uncertainty_actions(x)
    elif policy == "FAILURE_RISK_ONLY":
        action = np.where(p_failure >= .45, 2, 0)
    elif policy == "STATE_ONLY":
        action = expected_actions(p_state, effects, costs)
    elif policy == "ATTRIBUTION_AWARE":
        action = expected_actions(p_full, effects, costs)
    elif policy == "ATTRIBUTION_DEGRADED":
        action = expected_actions(p_degraded, effects, costs)
    elif policy == "SHUFFLED_ATTRIBUTION":
        action = expected_actions(p_shuffled, effects, costs)
    elif policy == "ORACLE_ATTRIBUTION":
        action = expected_actions(np.eye(5)[source], effects, costs)
    else:
        raise ValueError(f"unknown policy: {policy}")
    success = draw[np.arange(len(source)), action] < effects[action, source]
    intervention_cost = costs[action]
    wrong = (source != 0) & (action != source)
    unnecessary = (source == 0) & (action != 0)
    row = {
        "policy": policy,
        "task_success": float(success.mean()),
        "mean_intervention_cost": float(intervention_cost.mean()),
        "cost_adjusted_utility": float(success.mean() - intervention_cost.mean()),
        "wrong_intervention_rate": float(wrong.mean()),
        "unnecessary_intervention_rate": float(unnecessary.mean()),
    }
    for index, name in enumerate(ACTIONS):
        row[f"action_{name}_rate"] = float((action == index).mean())
    return row, action


def attribution_metrics(name: str, source: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    return {
        "scenario": name,
        "model": "full_attribution",
        "accuracy": float(accuracy_score(source, predicted)),
        "macro_f1": float(f1_score(source, predicted, average="macro")),
        "selector_f1": float(f1_score(source == 2, predicted == 2)),
        "selector_auroc": float(roc_auc_score(source == 2, probabilities[:, 2])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_x, _, calibration_source = episodes(8000, 2201)
    validation_x, validation_state, validation_source = episodes(2000, 2202)
    id_x, id_state, id_source = episodes(12000, 2203)
    ood_x, ood_state, ood_source = sample_ood()
    del validation_x, validation_state, validation_source  # Materialized only to uphold frozen split design.

    full = RandomForestClassifier(n_estimators=160, min_samples_leaf=8, n_jobs=-1, random_state=2201, class_weight="balanced_subsample").fit(calibration_x, calibration_source)
    state_only = RandomForestClassifier(n_estimators=160, min_samples_leaf=8, n_jobs=-1, random_state=2204).fit(calibration_x[:, :4], calibration_source)
    failure_risk = RandomForestClassifier(n_estimators=160, min_samples_leaf=8, n_jobs=-1, random_state=2207, class_weight="balanced_subsample").fit(calibration_x, calibration_source != 0)
    prior = np.bincount(calibration_source, minlength=5) / len(calibration_source)

    scenarios = {
        "ID": (id_x, id_state, id_source),
        "OOD_PREVALENCE": (ood_x, ood_state, ood_source),
    }
    result_rows: list[dict] = []
    confusion_rows: list[dict] = []
    metric_rows: list[dict] = []
    for scenario, (x, state, source) in scenarios.items():
        p_full = proba(full, x)
        p_state = proba(state_only, x[:, :4])
        p_failure = failure_risk.predict_proba(x)[:, list(failure_risk.classes_).index(True)]
        p_degraded = .5 * p_full + .5 * prior
        p_shuffled = p_full[np.random.default_rng(2206).permutation(len(p_full))]
        metric_rows.append(attribution_metrics(scenario, source, p_full))
        metric_rows.append({"scenario": scenario, "model": "state_only", "accuracy": float(accuracy_score(source, p_state.argmax(axis=1))), "macro_f1": float(f1_score(source, p_state.argmax(axis=1), average="macro")), "selector_f1": float(f1_score(source == 2, p_state.argmax(axis=1) == 2)), "selector_auroc": float(roc_auc_score(source == 2, p_state[:, 2]))})
        metric_rows.append({"scenario": scenario, "model": "failure_risk", "accuracy": None, "macro_f1": None, "selector_f1": None, "selector_auroc": float(roc_auc_score(source != 0, p_failure))})
        draw = np.random.default_rng(2212 if scenario == "ID" else 2213).random((len(source), len(ACTIONS)))
        configurations = [("BASE", "BASE")]
        if scenario == "ID":
            configurations = [(effect_name, cost_name) for effect_name in EFFECTS for cost_name in COSTS]
        for effect_name, cost_name in configurations:
            actions_by_policy: dict[str, np.ndarray] = {}
            for policy in POLICIES:
                row, actions = run_policy(policy, x, source, p_full, p_state, p_failure, p_degraded, p_shuffled, EFFECTS[effect_name], COSTS[cost_name], draw)
                row.update({"scenario": scenario, "effectiveness_regime": effect_name, "cost_regime": cost_name})
                result_rows.append(row)
                actions_by_policy[policy] = actions
            if scenario == "ID" and effect_name == "BASE" and cost_name == "BASE":
                for policy, actions in actions_by_policy.items():
                    for source_index, source_name in enumerate(SOURCES):
                        mask = source == source_index
                        for action_index, action_name in enumerate(ACTIONS):
                            confusion_rows.append({"policy": policy, "true_source": source_name, "action": action_name, "count": int(((actions == action_index) & mask).sum()), "rate_within_cause": float((actions[mask] == action_index).mean())})

    # Add regret and paired policy comparisons after all rows exist.
    for scenario in scenarios:
        combinations = {(row["effectiveness_regime"], row["cost_regime"]) for row in result_rows if row["scenario"] == scenario}
        for effect_name, cost_name in combinations:
            group = [row for row in result_rows if row["scenario"] == scenario and row["effectiveness_regime"] == effect_name and row["cost_regime"] == cost_name]
            by_policy = {row["policy"]: row for row in group}
            oracle = by_policy["ORACLE_ATTRIBUTION"]["cost_adjusted_utility"]
            best_simple = max(by_policy[name]["cost_adjusted_utility"] for name in ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY"))
            predicted = by_policy["ATTRIBUTION_AWARE"]["cost_adjusted_utility"]
            denominator = oracle - best_simple
            for row in group:
                row["regret_vs_oracle"] = oracle - row["cost_adjusted_utility"]
                row["attribution_minus_uncertainty"] = predicted - by_policy["UNCERTAINTY_ONLY"]["cost_adjusted_utility"]
                row["best_simple_utility"] = best_simple
                row["oracle_recovery_ratio"] = (predicted - best_simple) / denominator if denominator > 0 else None

    write_csv(OUT / "policy_results.csv", result_rows)
    write_csv(OUT / "intervention_confusion.csv", confusion_rows)
    write_csv(OUT / "attribution_metrics.csv", metric_rows)
    metadata = {
        "status": "PASS",
        "calibration_episodes": 8000,
        "validation_episodes_unused": 2000,
        "id_test_episodes": len(id_source),
        "ood_test_episodes": len(ood_source),
        "actions": ACTIONS,
        "costs": {name: values.tolist() for name, values in COSTS.items()},
        "effects": {name: values.tolist() for name, values in EFFECTS.items()},
        "full_attribution_id_macro_f1": metric_rows[0]["macro_f1"],
        "outputs": ["policy_results.csv", "intervention_confusion.csv", "attribution_metrics.csv"],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
