"""Stage-3 CPU-only multi-fault, marginal-attribution, budgeted-routing study."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
COMPONENTS = ("G", "V", "X", "E")
ACTIONS = ("COMMIT", "REGENERATE", "STRONG_VERIFY", "TOOL_RETRY", "REOBSERVE_REPLAN")
POLICIES = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY", "ATTRIBUTION_AWARE", "SHUFFLED_ATTRIBUTION", "ORACLE_MULTI_LABEL")
FEATURES = ("state_A", "state_B", "state_C", "state_D", "candidate_count", "margin", "entropy", "disagreement", "executor_status", "latency", "state_delta", "expected_transition", "environment_warning", "history")
CATEGORY_FAULTS = np.array([
    [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
], dtype=int)
ID_STATE_P = np.array([.25, .25, .30, .20])
ID_CATEGORY_P = np.array([
    [.50,.08,.06,.07,.07,.04,.03,.03,.03,.03,.06],
    [.32,.10,.10,.10,.10,.05,.05,.05,.04,.04,.05],
    [.25,.12,.15,.07,.10,.08,.04,.04,.04,.05,.06],
    [.30,.06,.08,.16,.10,.03,.05,.04,.06,.05,.07],
])
OOD_STATE_P = np.array([.15,.20,.35,.30])
OOD_CATEGORY_P = np.array([
    [.30,.07,.07,.07,.07,.08,.06,.06,.06,.06,.10],
    [.20,.08,.10,.09,.08,.08,.08,.08,.07,.06,.08],
    [.15,.07,.13,.06,.07,.15,.06,.06,.07,.08,.10],
    [.20,.04,.06,.14,.08,.05,.08,.06,.10,.08,.11],
])
EFFECTS = {
    "BASE": np.array([[.00,.00,.00,.00],[.70,.20,.15,.15],[.18,.75,.10,.12],[.12,.12,.70,.42],[.10,.12,.40,.75]]),
    "V_STRONG": np.array([[.00,.00,.00,.00],[.70,.20,.15,.15],[.18,.92,.10,.12],[.12,.12,.70,.42],[.10,.12,.40,.75]]),
}
RETENTION = np.array([.990,.980,.990,.985,.985])
COSTS = {
    "LOW": np.array([.00,.03,.08,.05,.07]),
    "MEDIUM": np.array([.00,.06,.16,.10,.12]),
    "HIGH": np.array([.00,.12,.28,.18,.22]),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def ece(probability: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo in np.linspace(0, .9, bins):
        high = lo + .1
        mask = (probability >= lo) & (probability < (high if high < 1 else 1.000001))
        if mask.any(): total += mask.mean() * abs(probability[mask].mean() - truth[mask].mean())
    return float(total)


def episodes(n: int, seed: int, state_p: np.ndarray = ID_STATE_P, category_p: np.ndarray = ID_CATEGORY_P) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """No latent component is emitted in x; y is held separately for supervision/evaluation."""
    rng = np.random.default_rng(seed)
    state = rng.choice(4, n, p=state_p)
    category = np.array([rng.choice(len(CATEGORY_FAULTS), p=category_p[s]) for s in state])
    faults = CATEGORY_FAULTS[category]
    g, v, x, e = faults.T
    count = np.clip(rng.normal(3.5 + .95*g + .35*v + .55*x + .25*e, .85), 1, 7)
    margin = np.clip(rng.normal(.68 - .22*g - .25*v - .10*x - .12*e, .16), .01, .99)
    entropy = np.clip(rng.normal(.28 + .28*g + .31*v + .18*x + .20*e, .17), .01, 1.3)
    disagreement = np.clip(rng.normal(.18 + .24*g + .35*v + .10*x + .14*e, .18), 0, 1)
    executor = np.clip(rng.normal(.10 + .10*g + .08*v + .55*x + .34*e, .19), 0, 1)
    latency = np.clip(rng.normal(.35 + .10*g + .06*v + .47*x + .41*e, .22), .02, 2)
    delta = np.clip(rng.normal(.15 + .30*g + .06*v + .13*x + .31*e, .18), 0, 1)
    transition = (rng.random(n) < np.clip(.88 - .39*g - .23*v - .16*x - .30*e, .03, .97)).astype(float)
    warning = (rng.random(n) < np.clip(.06 + .07*g + .08*v + .25*x + .50*e, .02, .98)).astype(float)
    history = np.clip(rng.normal(np.cumsum(faults.sum(axis=1) > 0) / (np.arange(n) + 1), .09), 0, 1)
    x_obs = np.column_stack([np.eye(4)[state], count, margin, entropy, disagreement, executor, latency, delta, transition, warning, history])
    return x_obs, state, faults, category


def fit_component_models(x: np.ndarray, faults: np.ndarray) -> list[RandomForestClassifier]:
    return [RandomForestClassifier(n_estimators=160, min_samples_leaf=10, n_jobs=-1, random_state=3101 + component, class_weight="balanced_subsample").fit(x, faults[:, component]) for component in range(4)]


def component_probabilities(models: list[RandomForestClassifier], x: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict_proba(x)[:, list(model.classes_).index(1)] for model in models])


def action_sequences(budget: int) -> tuple[tuple[int, ...], ...]:
    actions = tuple(range(1, len(ACTIONS)))
    return ((),) + tuple(sequence for length in range(1, budget + 1) for sequence in itertools.combinations(actions, length))


def sequence_terms(sequence: tuple[int, ...], effects: np.ndarray, costs: np.ndarray) -> tuple[np.ndarray, float, float]:
    if not sequence: return np.zeros(4), .990, 0.0
    repair = 1 - np.prod([1 - effects[action] for action in sequence], axis=0)
    retention = .990 * float(np.prod([RETENTION[action] for action in sequence]))
    return repair, retention, float(costs[list(sequence)].sum())


def select_expected(probabilities: np.ndarray, budget: int, effects: np.ndarray, costs: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    sequences = action_sequences(budget)
    scores = []
    for sequence in sequences:
        repair, retention, cost = sequence_terms(sequence, effects, costs)
        expected_success = retention * np.prod((1 - probabilities) + probabilities * repair, axis=1)
        scores.append(expected_success - cost)
    return np.argmax(np.column_stack(scores), axis=1), sequences


def generic_sequences(probability: np.ndarray, budget: int, uncertainty: bool) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    # Frozen generic routing: only confidence/risk controls a generic strong/tool portfolio.
    sequences = ((), (2,), (2, 3)) if budget == 2 else ((), (2,))
    medium = .58 if uncertainty else .45
    high = .78 if uncertainty else .70
    chosen = np.zeros(len(probability), dtype=int)
    chosen[probability >= medium] = 1
    if budget == 2: chosen[probability >= high] = 2
    return chosen, sequences


def action_outcome(faults: np.ndarray, indices: np.ndarray, sequences: tuple[tuple[int, ...], ...], effects: np.ndarray, costs: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success_probability = np.zeros(len(faults)); action_cost = np.zeros(len(faults)); repair_matrix = np.zeros((len(faults), 4))
    for choice, sequence in enumerate(sequences):
        mask = indices == choice
        if not mask.any(): continue
        repair, retention, cost = sequence_terms(sequence, effects, costs)
        repair_matrix[mask] = repair
        success_probability[mask] = retention * np.prod(np.where(faults[mask] == 1, repair, 1), axis=1)
        action_cost[mask] = cost
    success = draws[np.arange(len(faults)), indices] < success_probability
    return success, action_cost, repair_matrix


def run_policy(policy: str, x: np.ndarray, faults: np.ndarray, p_full: np.ndarray, p_state: np.ndarray, risk: np.ndarray, p_shuffled: np.ndarray, budget: int, effects: np.ndarray, costs: np.ndarray, draws: np.ndarray) -> tuple[dict, np.ndarray, tuple[tuple[int, ...], ...]]:
    if policy == "COMMIT_ONLY":
        sequences = ((),); choice = np.zeros(len(faults), dtype=int)
    elif policy == "ALWAYS_STRONG_VERIFY":
        sequences = ((2,),); choice = np.zeros(len(faults), dtype=int)
    elif policy == "UNCERTAINTY_ONLY":
        uncertainty = ((1 - x[:, 5]) + x[:, 6] / 1.3 + x[:, 7]) / 3
        choice, sequences = generic_sequences(uncertainty, budget, True)
    elif policy == "FAILURE_RISK_ONLY":
        choice, sequences = generic_sequences(risk, budget, False)
    elif policy == "STATE_ONLY":
        choice, sequences = select_expected(p_state, budget, effects, costs)
    elif policy == "ATTRIBUTION_AWARE":
        choice, sequences = select_expected(p_full, budget, effects, costs)
    elif policy == "SHUFFLED_ATTRIBUTION":
        choice, sequences = select_expected(p_shuffled, budget, effects, costs)
    elif policy == "ORACLE_MULTI_LABEL":
        choice, sequences = select_expected(faults.astype(float), budget, effects, costs)
    else: raise ValueError(policy)
    success, action_cost, repair = action_outcome(faults, choice, sequences, effects, costs, draws)
    active = faults.sum(axis=1) > 0
    selected_action_mask = np.zeros((len(faults), 4), dtype=bool)
    for index, sequence in enumerate(sequences):
        for action in sequence: selected_action_mask[choice == index, action - 1] = True
    # An intervention is wrong only if it offers no moderate repair chance to any active component.
    has_intervention = np.array([bool(sequences[index]) for index in choice])
    wrong = active & has_intervention & (np.max(np.where(faults == 1, repair, 0), axis=1) < .40)
    unnecessary = (~active) & has_intervention
    row = {"policy": policy, "budget": budget, "task_success": float(success.mean()), "mean_intervention_cost": float(action_cost.mean()), "cost_adjusted_utility": float(success.mean() - action_cost.mean()), "wrong_intervention_rate": float(wrong.mean()), "unnecessary_intervention_rate": float(unnecessary.mean())}
    for action, name in enumerate(ACTIONS[1:], start=0): row[f"action_{name}_rate"] = float(selected_action_mask[:, action].mean())
    row["mean_actions_used"] = float(selected_action_mask.sum(axis=1).mean())
    return row, choice, sequences


def multilabel_metrics(scenario: str, faults: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    rows = []
    predicted = probabilities >= .5
    for index, component in enumerate(COMPONENTS):
        rows.append({"scenario": scenario, "component": component, "auroc": float(roc_auc_score(faults[:, index], probabilities[:, index])), "f1": float(f1_score(faults[:, index], predicted[:, index])), "brier": float(np.mean((probabilities[:, index] - faults[:, index]) ** 2)), "ece": ece(probabilities[:, index], faults[:, index])})
    rows.append({"scenario": scenario, "component": "aggregate", "auroc": None, "f1": float(f1_score(faults, predicted, average="macro")), "brier": float(np.mean((probabilities - faults) ** 2)), "ece": float(np.mean([ece(probabilities[:, index], faults[:, index]) for index in range(4)])), "micro_f1": float(f1_score(faults, predicted, average="micro")), "exact_match": float(np.all(faults == predicted, axis=1).mean())})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_x, calibration_state, calibration_faults, _ = episodes(10000, 3101)
    validation_x, validation_state, validation_faults, _ = episodes(2500, 3102)
    id_x, id_state, id_faults, _ = episodes(15000, 3103)
    ood_x, ood_state, ood_faults, _ = episodes(15000, 3104, OOD_STATE_P, OOD_CATEGORY_P)
    del validation_x, validation_state, validation_faults
    full_models = fit_component_models(calibration_x, calibration_faults)
    state_models = fit_component_models(calibration_x[:, :4], calibration_faults)
    risk_model = RandomForestClassifier(n_estimators=160, min_samples_leaf=10, n_jobs=-1, random_state=3110, class_weight="balanced_subsample").fit(calibration_x, calibration_faults.sum(axis=1) > 0)
    scenarios = {"ID": (id_x, id_faults), "OOD_DUAL_SHIFT": (ood_x, ood_faults)}
    result_rows: list[dict] = []; metric_rows: list[dict] = []; confusion_rows: list[dict] = []
    for scenario, (x, faults) in scenarios.items():
        p_full = component_probabilities(full_models, x); p_state = component_probabilities(state_models, x[:, :4])
        risk = risk_model.predict_proba(x)[:, list(risk_model.classes_).index(True)]
        p_shuffled = p_full[np.random.default_rng(3106).permutation(len(p_full))]
        metric_rows.extend(multilabel_metrics(scenario, faults, p_full))
        draw = np.random.default_rng(3112 if scenario == "ID" else 3113).random((len(faults), len(action_sequences(2))))
        regimes = [("BASE", name, budget) for name in COSTS for budget in (1, 2)] if scenario == "ID" else [("BASE", "MEDIUM", 2)]
        regimes += [("V_STRONG", name, budget) for name in COSTS for budget in (1, 2)] if scenario == "ID" else []
        for effect_name, cost_name, budget in regimes:
            actions = {}
            for policy in POLICIES:
                row, choice, sequences = run_policy(policy, x, faults, p_full, p_state, risk, p_shuffled, budget, EFFECTS[effect_name], COSTS[cost_name], draw)
                row.update({"scenario": scenario, "effectiveness_regime": effect_name, "cost_regime": cost_name})
                result_rows.append(row); actions[policy] = (choice, sequences)
            if scenario == "ID" and effect_name == "BASE" and cost_name == "MEDIUM" and budget == 1:
                for policy, (choice, sequences) in actions.items():
                    for component, component_name in enumerate(COMPONENTS):
                        mask = faults[:, component] == 1
                        for action in range(1, len(ACTIONS)):
                            used = np.array([action in sequences[index] for index in choice])
                            confusion_rows.append({"policy": policy, "true_component": component_name, "action": ACTIONS[action], "rate_within_component": float(used[mask].mean()), "count": int((used & mask).sum())})
    for scenario in scenarios:
        combinations = {(r["effectiveness_regime"], r["cost_regime"], r["budget"]) for r in result_rows if r["scenario"] == scenario}
        for effect, cost, budget in combinations:
            group = [r for r in result_rows if r["scenario"] == scenario and r["effectiveness_regime"] == effect and r["cost_regime"] == cost and r["budget"] == budget]
            by = {r["policy"]: r for r in group}; oracle = by["ORACLE_MULTI_LABEL"]["cost_adjusted_utility"]
            simple_names = ("COMMIT_ONLY", "ALWAYS_STRONG_VERIFY", "UNCERTAINTY_ONLY", "FAILURE_RISK_ONLY", "STATE_ONLY")
            best_simple = max(by[name]["cost_adjusted_utility"] for name in simple_names); attribution = by["ATTRIBUTION_AWARE"]["cost_adjusted_utility"]
            denominator = oracle - best_simple
            for row in group:
                row["regret_vs_oracle"] = oracle - row["cost_adjusted_utility"]
                row["attribution_minus_uncertainty"] = attribution - by["UNCERTAINTY_ONLY"]["cost_adjusted_utility"]
                row["attribution_minus_best_simple"] = attribution - best_simple
                row["best_simple_utility"] = best_simple
                row["oracle_recovery_ratio"] = (attribution - best_simple) / denominator if denominator > 0 else None
    write_csv(OUT / "policy_results.csv", result_rows); write_csv(OUT / "component_action_confusion.csv", confusion_rows); write_csv(OUT / "multilabel_attribution_metrics.csv", metric_rows)
    metadata = {"status": "PASS", "calibration_episodes": 10000, "validation_episodes_unused": 2500, "id_test_episodes": 15000, "ood_test_episodes": 15000, "id_prevalence": {"none": float((id_faults.sum(axis=1)==0).mean()), "single": float((id_faults.sum(axis=1)==1).mean()), "dual": float((id_faults.sum(axis=1)==2).mean())}, "ood_prevalence": {"none": float((ood_faults.sum(axis=1)==0).mean()), "single": float((ood_faults.sum(axis=1)==1).mean()), "dual": float((ood_faults.sum(axis=1)==2).mean())}, "outputs": ["policy_results.csv", "component_action_confusion.csv", "multilabel_attribution_metrics.csv"]}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
