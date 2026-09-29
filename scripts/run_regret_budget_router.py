"""CPU-only, group-safe analysis for the D2 Regret 1024->4096 router.

``build`` consumes only frozen target-blind D2 traces and writes a hash-frozen
1024-state feature table.  ``score`` is deliberately a separate post-freeze
step: it attaches Gold only to define/evaluate the continuation label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.solution_normalization import normalize_arc_solutions
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.turbodfs_v4_common import sha256_file

EXPERIMENT = "REGRET_BUDGET_ROUTER_V1"
CUT = 1024
MAX_NODES = 4096
POLICY = "CUMULATIVE_REGRET_r=4.00"
VALIDATION_LABEL = "REGRET4_4096_VALIDATION"
FEATURE_CANDIDATES = (
    "candidate_yield_last256", "nodes_since_last_candidate",
    "frontier_growth_last256", "retained_successors_last256",
    "frontier_size_1024", "retained_regret_min_proxy_1024",
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (index - lo)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def parse_cell(key: str) -> tuple[str, int, int, str]:
    task, output, depth, view = key.split(":", 3)
    return task, int(output.removeprefix("o")), int(depth.removeprefix("d")), view


def raw_records(run: Path) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    manifest = read_json(run / "D2_BUDGET_MANIFEST.json")
    flag = read_json(run / "D2_BUDGET_GENERATION_FROZEN.flag")
    records: list[tuple[Path, dict[str, Any]]] = []
    actual: dict[str, str] = {}
    for path in sorted((run / "raw").rglob("*.json")):
        relative = path.relative_to(run).as_posix()
        actual[relative] = sha256_file(path)
        records.append((path, read_json(path)))
    if len(records) != int(flag["records"]) or actual != flag["raw_hashes"]:
        raise RuntimeError("frozen D2 raw hash validation failed")
    return manifest, records


def candidates_at(row: dict[str, Any], cutoff: int) -> list[dict[str, Any]]:
    ids = {
        int(event["candidate_completion_index"])
        for event in row["search_trace"]
        if event.get("candidate_completion_index") is not None
        and int(event.get("nodes_expanded_so_far") or 0) <= cutoff
    }
    return [candidate for candidate in row["candidates"] if int(candidate["candidate_id"]) in ids]


def window_events(events: list[dict[str, Any]], lower: int) -> list[dict[str, Any]]:
    return [event for event in events if int(event.get("nodes_expanded_so_far") or 0) > lower]


def node_stats(events: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    per_node: dict[int, tuple[float, float]] = {}
    for event in events:
        node = int(event.get("nodes_expanded_so_far") or 0)
        considered, retained = number(event.get("successors_considered")), number(event.get("successors_retained"))
        if node and considered is not None and retained is not None:
            old = per_node.get(node, (0.0, 0.0))
            per_node[node] = max(old[0], considered), max(old[1], retained)
    return mean(value[0] for value in per_node.values()), mean(value[1] for value in per_node.values())


def features_for(path: Path, raw_relative_path: str, row: dict[str, Any]) -> dict[str, Any]:
    key = row["d2_job"]["cell_key"]
    task, output_index, depth, view = parse_cell(key)
    events = [event for event in row["search_trace"] if int(event.get("nodes_expanded_so_far") or 0) <= CUT]
    before = candidates_at(row, CUT)
    last128, last256, last512 = (window_events(events, CUT - width) for width in (128, 256, 512))
    completed_by_window = lambda event_list: len({int(e["candidate_completion_index"]) for e in event_list if e.get("candidate_completion_index") is not None})
    completed = len(before)
    grids = [candidate.get("canonical_candidate") for candidate in before if candidate.get("valid_grid")]
    grid_keys = [digest(grid) for grid in grids]
    frontier = [number(e.get("frontier_size_at_insert")) for e in events] + [number(e.get("frontier_size_at_pop")) for e in events]
    frontier = [value for value in frontier if value is not None]
    frontier_128 = [number(e.get("frontier_size_at_insert")) for e in last128] + [number(e.get("frontier_size_at_pop")) for e in last128]
    frontier_256 = [number(e.get("frontier_size_at_insert")) for e in last256] + [number(e.get("frontier_size_at_pop")) for e in last256]
    frontier_128, frontier_256 = [v for v in frontier_128 if v is not None], [v for v in frontier_256 if v is not None]
    retained = [e for e in events if e.get("frontier_insert_order") is not None]
    regrets = [number(e.get("path_cumulative_regret")) for e in retained]
    regrets = [value for value in regrets if value is not None]
    prefixes = [number(e.get("prefix_length")) for e in retained]
    prefixes = [value for value in prefixes if value is not None]
    considered, retained_per_node = node_stats(events)
    _, retained_last256 = node_stats(last256)
    candidate_nodes = [int(e.get("nodes_expanded_so_far") or 0) for e in events if e.get("candidate_completion_index") is not None]
    last_candidate = max(candidate_nodes) if candidate_nodes else 0
    row_features = {
        "cell_key": key, "output_id": f"{task}:o{output_index}", "task_id": task,
        "output_index": output_index, "depth": depth, "view": view,
        # A cell filename is shared by D2's 4096 and prefix-budget jobs, so
        # basename lookup is not an identity.  Preserve the immutable path
        # relative to the frozen run for exact post-freeze Gold annotation.
        "raw_relative_path": raw_relative_path, "nodes_expanded_total": int(row["nodes_expanded"]),
        "termination_reason_total": row["termination_reason"],
        "additional_nodes_actual": min(max(0, int(row["nodes_expanded"]) - CUT), MAX_NODES - CUT),
        "candidate_count_1024": completed,
        "unique_candidate_count_1024": len(set(grid_keys)),
        "candidate_count_last128": completed_by_window(last128),
        "candidate_count_last256": completed_by_window(last256),
        "candidate_count_last512": completed_by_window(last512),
        "candidate_yield_total": completed / CUT,
        "candidate_yield_last256": completed_by_window(last256) / 256,
        "candidate_yield_last512": completed_by_window(last512) / 512,
        "nodes_since_last_candidate": CUT - last_candidate if candidate_nodes else CUT,
        "exact_duplicate_rate_proxy": 0.0 if not grid_keys else 1.0 - len(set(grid_keys)) / len(grid_keys),
        "unique_grid_ratio": (len(set(grid_keys)) / completed) if completed else 0.0,
        "frontier_size_1024": frontier[-1] if frontier else 0.0,
        "max_frontier_size_to_1024": max(frontier) if frontier else 0.0,
        "mean_frontier_size_to_1024": mean(frontier) or 0.0,
        "frontier_growth_last128": ((mean(frontier_128) or 0.0) - (mean(frontier[:-len(frontier_128)]) or 0.0)) if frontier_128 else 0.0,
        "frontier_growth_last256": ((mean(frontier_256) or 0.0) - (mean(frontier[:-len(frontier_256)]) or 0.0)) if frontier_256 else 0.0,
        "successors_considered_per_node": considered or 0.0,
        "successors_retained_per_node": retained_per_node or 0.0,
        "retained_successors_last256": retained_last256 or 0.0,
        "retained_regret_min_proxy_1024": min(regrets) if regrets else 0.0,
        "retained_regret_median_proxy_1024": quantile(regrets, .5) or 0.0,
        "retained_regret_p75_proxy_1024": quantile(regrets, .75) or 0.0,
        "retained_regret_p90_proxy_1024": quantile(regrets, .9) or 0.0,
        "retained_regret_spread_proxy_1024": (max(regrets) - min(regrets)) if regrets else 0.0,
        "mean_retained_prefix_length_proxy_1024": mean(prefixes) or 0.0,
        "max_retained_prefix_length_proxy_1024": max(prefixes) if prefixes else 0.0,
        "completed_candidate_mean_length": mean(len(candidate.get("candidate_token_ids", [])) for candidate in before) or 0.0,
        "termination_near_1024": int(int(row["nodes_expanded"]) <= CUT),
    }
    return row_features


def build(args: argparse.Namespace) -> None:
    run, out = args.d2_run.resolve(), args.out.resolve()
    manifest, rows = raw_records(run)
    features, early = [], []
    for path, row in rows:
        job = row.get("d2_job", {})
        if job.get("label") != VALIDATION_LABEL or row.get("decoder_policy") != POLICY:
            continue
        if int(row["nodes_expanded"]) < CUT:
            early.append({"cell_key": job["cell_key"], "output_id": row["output_id"], "nodes_expanded": row["nodes_expanded"], "termination_reason": row["termination_reason"], "reason": "no_1024_decision_state"})
            continue
        features.append(features_for(path, path.relative_to(run).as_posix(), row))
    features.sort(key=lambda row: row["cell_key"])
    write_csv(out / "router_features_frozen.csv", features)
    write_csv(out / "router_ineligible_early_termination.csv", early)
    schema = {
        "experiment_id": EXPERIMENT, "cutoff_nodes": CUT, "source": "frozen D2 REGRET4_4096_VALIDATION raw traces",
        "target_blind": True, "features_exclude": ["Gold rank", "Gold NLL", "Gold survival", "first_gold_node", "exact hit", "future 4096 statistics"],
        "feature_notes": {"retained_regret_*_proxy_1024": "retained trace paths, not an exactly reconstructible active frontier", "exact_duplicate_rate_proxy": "completed valid candidate grids at/before node 1024", "pairwise_hamming": "not emitted: complete per-candidate grids are insufficiently standardized for an unambiguous cheap metric"},
        "metadata": ["depth", "view", "output_id"], "model_features": list(FEATURE_CANDIDATES),
        "ineligible_before_1024": len(early), "router_rows": len(features), "raw_manifest_sha256": sha256_file(run / "D2_BUDGET_MANIFEST.json"),
    }
    dump(out / "feature_schema.json", schema)
    flag = {"experiment_id": EXPERIMENT, "router_rows": len(features), "features_sha256": sha256_file(out / "router_features_frozen.csv"), "d2_raw_hashes_verified": True, "solutions_accessed": False, "source_manifest": manifest["challenge_sha256"]}
    dump(out / "ROUTER_FEATURES_FROZEN.json", flag)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_value(row: dict[str, Any], name: str) -> float:
    return float(row[name])


def first_gold_node(raw: dict[str, Any], gold: list[list[int]]) -> int | None:
    gold_ids = {int(c["candidate_id"]) for c in raw["candidates"] if c.get("valid_grid") and c.get("canonical_candidate") == gold}
    values = [int(e.get("nodes_expanded_so_far") or 0) for e in raw["search_trace"] if e.get("candidate_completion_index") in gold_ids]
    return min(values) if values else None


def point_biserial(rows: list[dict[str, Any]], feature: str) -> float | None:
    xs, ys = [feature_value(r, feature) for r in rows], [int(r["Y_CONTINUE"]) for r in rows]
    if len(set(xs)) < 2 or not all(ys) and not any(ys):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def threshold_fit(train: list[dict[str, Any]], feature: str) -> tuple[str, float]:
    positives = [r for r in train if int(r["Y_CONTINUE"])]
    if not positives:
        return "always", 0.0
    values = sorted(set(feature_value(r, feature) for r in train))
    qs = sorted(set(v for q in (0., .25, .5, .75, 1.) if (v := quantile(values, q)) is not None))
    candidates: list[tuple[tuple[float, float], str, float]] = []
    for direction in ("gt", "le"):
        for threshold in qs:
            pred = [feature_value(r, feature) > threshold if direction == "gt" else feature_value(r, feature) <= threshold for r in train]
            recall = sum(p and int(r["Y_CONTINUE"]) for p, r in zip(pred, train)) / len(positives)
            rate = sum(pred) / len(pred)
            candidates.append(((-recall, rate), direction, threshold))
    return min(candidates, key=lambda item: item[0])[1:]


def standardize(train: list[dict[str, Any]], names: tuple[str, ...]) -> tuple[list[float], list[float]]:
    means = [statistics.fmean(feature_value(row, name) for row in train) for name in names]
    scales = [math.sqrt(statistics.fmean((feature_value(row, name) - mu) ** 2 for row in train)) or 1.0 for name, mu in zip(names, means)]
    return means, scales


def logistic_fit(train: list[dict[str, Any]], names: tuple[str, ...], C: float) -> tuple[list[float], list[float], list[float], float]:
    means, scales = standardize(train, names); weights = [0.0] * len(names); bias = 0.0
    if len({int(r["Y_CONTINUE"]) for r in train}) == 1:
        return means, scales, weights, 20.0 if int(train[0]["Y_CONTINUE"]) else -20.0
    for _ in range(400):
        grad, gb = [0.0] * len(names), 0.0
        for row in train:
            values = [(feature_value(row, name) - mu) / scale for name, mu, scale in zip(names, means, scales)]
            z = max(-30.0, min(30.0, bias + sum(w * x for w, x in zip(weights, values))))
            p = 1.0 / (1.0 + math.exp(-z)); delta = p - int(row["Y_CONTINUE"])
            gb += delta
            for i, value in enumerate(values): grad[i] += delta * value
        n = len(train)
        for i in range(len(weights)): weights[i] -= .25 * ((grad[i] / n) + weights[i] / C)
        bias -= .25 * gb / n
    return means, scales, weights, bias


def logistic_predict(row: dict[str, Any], names: tuple[str, ...], model: tuple[list[float], list[float], list[float], float]) -> float:
    means, scales, weights, bias = model
    z = max(-30.0, min(30.0, bias + sum(w * ((feature_value(row, name) - mu) / scale) for name, mu, scale, w in zip(names, means, scales, weights))))
    return 1.0 / (1.0 + math.exp(-z))


def gini(rows: list[dict[str, Any]]) -> float:
    if not rows: return 0.0
    p = sum(int(r["Y_CONTINUE"]) for r in rows) / len(rows)
    return 2 * p * (1 - p)


def tree_fit(rows: list[dict[str, Any]], names: tuple[str, ...], depth: int) -> Any:
    if depth == 0 or len(rows) < 4 or len({int(r["Y_CONTINUE"]) for r in rows}) == 1:
        return ("leaf", sum(int(r["Y_CONTINUE"]) for r in rows) / len(rows))
    parent = gini(rows); best: tuple[float, str, float, list[dict[str, Any]], list[dict[str, Any]]] | None = None
    for name in names:
        values = sorted(set(feature_value(r, name) for r in rows))
        for threshold in sorted(set(quantile(values, q) for q in (.25, .5, .75))):
            left, right = [r for r in rows if feature_value(r, name) <= threshold], [r for r in rows if feature_value(r, name) > threshold]
            if len(left) < 2 or len(right) < 2: continue
            gain = parent - (len(left) * gini(left) + len(right) * gini(right)) / len(rows)
            if best is None or gain > best[0] or (gain == best[0] and (name, threshold) < (best[1], best[2])): best = (gain, name, threshold, left, right)
    if best is None or best[0] <= 0: return ("leaf", sum(int(r["Y_CONTINUE"]) for r in rows) / len(rows))
    _, name, threshold, left, right = best
    return ("split", name, threshold, tree_fit(left, names, depth - 1), tree_fit(right, names, depth - 1))


def tree_predict(row: dict[str, Any], tree: Any) -> float:
    while tree[0] == "split": tree = tree[3] if feature_value(row, tree[1]) <= tree[2] else tree[4]
    return float(tree[1])


def output_splits(rows: list[dict[str, Any]], scheme: str) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    groups = sorted({r["output_id"] for r in rows})
    if scheme == "loo": return [(group, [r for r in rows if r["output_id"] != group], [r for r in rows if r["output_id"] == group]) for group in groups]
    buckets: dict[int, set[str]] = defaultdict(set)
    for group in groups: buckets[int(hashlib.sha256(group.encode()).hexdigest(), 16) % 4].add(group)
    return [(f"sha256_mod4={fold}", [r for r in rows if r["output_id"] not in held], [r for r in rows if r["output_id"] in held]) for fold, held in sorted(buckets.items())]


def predictions(rows: list[dict[str, Any]], scheme: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fold, train, test in output_splits(rows, scheme):
        models: list[tuple[str, str, Any]] = [("ALWAYS_STOP", "fixed", None), ("ALWAYS_CONTINUE", "fixed", None)]
        for feature in FEATURE_CANDIDATES: models.append((f"THRESHOLD_{feature}", "threshold", feature))
        for C in (.1, 1., 10.): models.append((f"LOGISTIC_C={C:g}", "logistic", C))
        for depth in (1, 2, 3): models.append((f"TREE_DEPTH={depth}", "tree", depth))
        for name, kind, parameter in models:
            if kind == "fixed": fit = None
            elif kind == "threshold": fit = threshold_fit(train, parameter)
            elif kind == "logistic": fit = logistic_fit(train, FEATURE_CANDIDATES, float(parameter))
            else: fit = tree_fit(train, FEATURE_CANDIDATES, int(parameter))
            for row in test:
                if name == "ALWAYS_STOP": score = 0.0
                elif name == "ALWAYS_CONTINUE": score = 1.0
                elif kind == "threshold":
                    direction, threshold = fit; score = 1.0 if direction == "always" or (feature_value(row, parameter) > threshold if direction == "gt" else feature_value(row, parameter) <= threshold) else 0.0
                elif kind == "logistic": score = logistic_predict(row, FEATURE_CANDIDATES, fit)
                else: score = tree_predict(row, fit)
                result.append({"cv_scheme": scheme, "fold": fold, "model": name, "output_id": row["output_id"], "cell_key": row["cell_key"], "Y_CONTINUE": row["Y_CONTINUE"], "score": score, "additional_nodes_actual": row["additional_nodes_actual"]})
    return result


def operating(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    positives = [r for r in rows if int(r["Y_CONTINUE"])]
    total_cost = sum(float(r["additional_nodes_actual"]) for r in rows)
    thresholds = sorted({float(r["score"]) for r in rows} | {-1.0}, reverse=True)
    options = []
    for threshold in thresholds:
        pred = [float(r["score"]) >= threshold for r in rows]
        recall = sum(p and int(r["Y_CONTINUE"]) for p, r in zip(pred, rows)) / len(positives) if positives else 1.0
        continuation = sum(pred) / len(pred)
        saved = sum(float(r["additional_nodes_actual"]) for p, r in zip(pred, rows) if not p)
        if recall >= target: options.append((continuation, -saved, threshold, recall, saved, pred))
    if not options: raise RuntimeError("always-continue threshold missing")
    continuation, negative_saved, threshold, recall, saved, pred = min(options)
    return {"threshold": threshold, "late_rescue_recall": recall, "continuation_rate": continuation, "estimated_extra_nodes_saved": saved, "extra_node_saving_fraction": saved / total_cost if total_cost else 0.0, "false_stop_count": sum((not p) and int(r["Y_CONTINUE"]) for p, r in zip(pred, rows)), "false_continue_count": sum(p and not int(r["Y_CONTINUE"]) for p, r in zip(pred, rows))}


def expansion_plan(rows: list[dict[str, Any]], repo: Path, out: Path) -> tuple[int, float]:
    source = repo / "analysis" / "decoder_pruning_counterfactual_v1" / "pruning_policy_cells.csv"
    known = {r["output_id"] for r in rows}; eligible: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(source):
        if row["policy"] != "RELREGRET_4.0" or row["output_id"] in known: continue
        # Deliberately read only identifiers/depth/view.  Gold-derived columns in
        # this historical table are never consulted for eligibility or ranking.
        eligible[(row["depth"], row["view"])].append({key: row[key] for key in ("task_id", "output_index", "output_id", "depth", "view")})
    chosen = []
    for stratum in sorted(eligible):
        ordered = sorted(eligible[stratum], key=lambda row: hashlib.sha256(f"{EXPERIMENT}/expansion/{row['output_id']}:d{row['depth']}:{row['view']}".encode()).hexdigest())
        chosen.extend(ordered[:5])
    chosen = sorted(chosen, key=lambda row: hashlib.sha256(f"{EXPERIMENT}/expansion/{row['output_id']}:d{row['depth']}:{row['view']}".encode()).hexdigest())
    runtime_csv = repo / "analysis" / "regret_budget_and_retrieval_v1" / "regret_runtime_analysis.csv"
    runtime = [float(r["runtime_seconds"]) for r in read_csv(runtime_csv) if r["label"] == VALIDATION_LABEL]
    estimate = statistics.median(runtime) * len(chosen) if runtime else 0.0
    lines = ["# Regret router deterministic expansion plan", "", "This is a plan only: no GPU work was launched.", "", f"- Proposed rows: {len(chosen)} (five SHA256-ranked cells in each available depth/view stratum).", "- Eligibility reads only frozen identifier/depth/view fields from D0 RELREGRET_4.0 rows; Gold-derived columns are not read.", f"- Existing router output IDs excluded: {len(known)}.", f"- Estimated GPU time: {estimate / 3600:.2f} GPU-hours using the D2 validation-cell median runtime; about {estimate / 2 / 3600:.2f} h wall on two equal GPUs, before overhead.", "", "## Frozen proposed cells", ""]
    for row in chosen: lines.append(f"- `{row['output_id']}:d{row['depth']}:{row['view']}`")
    (out / "REGRET_ROUTER_DATA_EXPANSION_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(out / "REGRET_ROUTER_DATA_EXPANSION_COHORT.json", {"experiment_id": EXPERIMENT, "selection": "SHA256(stratum-balanced; identifiers only)", "rows": chosen, "estimated_gpu_seconds": estimate})
    return len(chosen), estimate


def pause_resume_audit(repo: Path, out: Path) -> None:
    text = """# Regret pause/resume static audit

## Result

- PAUSE_RESUME_FEASIBLE: NO for the current public D1 TurboDFS interface.
- PAUSE_RESUME_PARITY: NOT_RUN (no GPU execution authorized or needed for this static result).

`inference_d1_turbo_dfs` creates a new local state dictionary and root KV cache for every call. The recursive `d1_turbo_dfs` owns the frontier candidate lists, prefixes, scores/regrets, parent nodes, `past_key_values`, counters, dedup/trace-pending state, and clock state only as call-local values. There is no public checkpoint/resume object or continuation API.

An implementation would need to retain, in one live process: frontier paths and order, candidate pool/dedup state, prefix tokens, path NLL/regret, parent node IDs, counters, termination/budget state, trace state, and the current KV cache. Disk serialization of KV is not required for an immediate in-process router decision, but parity must be proven before deployment.
"""
    (out / "REGRET_PAUSE_RESUME_AUDIT.md").write_text(text, encoding="utf-8")


def score(args: argparse.Namespace) -> None:
    run, out, repo = args.d2_run.resolve(), args.out.resolve(), ROOT
    frozen = read_json(out / "ROUTER_FEATURES_FROZEN.json")
    if sha256_file(out / "router_features_frozen.csv") != frozen["features_sha256"]: raise RuntimeError("router feature freeze hash mismatch")
    manifest, raw = raw_records(run)
    raw_by_relative = {path.relative_to(run).as_posix(): row for path, row in raw}
    if len(raw_by_relative) != len(raw): raise RuntimeError("non-unique frozen raw relative paths")
    challenge = read_json(Path(manifest["challenge_path"])); challenge_ids = list(challenge)
    counts = {task: len(item["test"]) for task, item in challenge.items()}
    solutions = normalize_arc_solutions(read_json(args.solutions), task_ids_in_challenge_order=challenge_ids, expected_output_counts=counts)
    dataset = []
    for row in read_csv(out / "router_features_frozen.csv"):
        raw_path = row["raw_relative_path"]
        if raw_path not in raw_by_relative: raise RuntimeError(f"frozen raw record missing: {raw_path}")
        raw_row = raw_by_relative[raw_path]
        if raw_row["d2_job"].get("cell_key") != row["cell_key"] or raw_row["d2_job"].get("label") != VALIDATION_LABEL:
            raise RuntimeError(f"frozen raw identity mismatch for {row['cell_key']}: {raw_path}")
        gold = solutions[row["task_id"]][int(row["output_index"])]
        first = first_gold_node(raw_row, gold)
        labelled = dict(row); labelled["first_gold_node_EVAL_ONLY"] = first; labelled["Y_CONTINUE"] = int(first is not None and CUT < first <= MAX_NODES)
        dataset.append(labelled)
    write_csv(out / "router_dataset.csv", dataset)
    labels = {"Y_CONTINUE": "1 iff Gold absent by node 1024 and first appears at nodes 1025..4096; Gold is evaluation-only and never a feature.", "rows": len(dataset), "positive": sum(int(r["Y_CONTINUE"]) for r in dataset), "negative": sum(not int(r["Y_CONTINUE"]) for r in dataset)}
    dump(out / "label_definition.json", labels)
    univariate = [{"feature": feature, "point_biserial_full_development_only": point_biserial(dataset, feature)} for feature in FEATURE_CANDIDATES]
    write_csv(out / "feature_univariate_analysis.csv", sorted(univariate, key=lambda r: abs(r["point_biserial_full_development_only"] or 0), reverse=True))
    predictions_all = predictions(dataset, "loo") + predictions(dataset, "sha256_mod4")
    write_csv(out / "grouped_cv_predictions.csv", predictions_all)
    summary = []
    for scheme in ("loo", "sha256_mod4"):
        for model in sorted({r["model"] for r in predictions_all if r["cv_scheme"] == scheme}):
            model_rows = [r for r in predictions_all if r["cv_scheme"] == scheme and r["model"] == model]
            for target, label in ((1.0, "100"), (.9, "90"), (.8, "80")):
                summary.append({"cv_scheme": scheme, "model": model, "operating_point": label, **operating(model_rows, target)})
    write_csv(out / "operating_points.csv", summary)
    write_csv(out / "router_model_summary.csv", summary)
    loo100 = [r for r in summary if r["cv_scheme"] == "loo" and r["operating_point"] == "100"]
    best = min(loo100, key=lambda r: (r["continuation_rate"], -r["estimated_extra_nodes_saved"], r["model"]))
    features_ranked = [r["feature"] for r in sorted(univariate, key=lambda r: abs(r["point_biserial_full_development_only"] or 0), reverse=True)]
    expansion_rows, estimated_seconds = expansion_plan(dataset, repo, out); pause_resume_audit(repo, out)
    outputs = sorted({r["output_id"] for r in dataset}); positive_outputs = sorted({r["output_id"] for r in dataset if int(r["Y_CONTINUE"])})
    status = "ROUTER_SIGNAL_NOT_ESTABLISHED" if labels["positive"] < 8 or len(outputs) < 20 else "ROUTER_SIGNAL_WEAK"
    decision = {"experiment_id": EXPERIMENT, "scope": "NONBLIND DEVELOPMENT; historical 33/89 unchanged", "router_rows": len(dataset), "unique_outputs": len(outputs), "positive": labels["positive"], "negative": labels["negative"], "positive_unique_outputs": len(positive_outputs), "best_single_feature": features_ranked[0] if features_ranked else None, "best_simple_router": best["model"], "loo_best_100": best, "router_signal": status, "more_data_required": True, "proposed_expansion_rows": expansion_rows, "estimated_gpu_seconds": estimated_seconds, "pause_resume_feasible": "NO", "pause_resume_parity": "NOT_RUN", "next": "EXPAND_ROUTER_DATA", "gpu_used": "NO", "two_feature_rules": "NOT_RUN: no preregistered R0 relationship and insufficient grouped sample"}
    dump(out / "REGRET_BUDGET_ROUTER_DECISION.json", decision)
    report = ["# Regret budget router v1", "", "CPU-only, post-freeze nonblind development analysis. Historical union remains 33/89 unchanged.", "", f"- Router rows at node 1024: {len(dataset)}; early-terminated D2 validation cells excluded: {frozen['router_rows'] + len(read_csv(out / 'router_ineligible_early_termination.csv')) - len(dataset)}.", f"- Unique outputs: {len(outputs)}; Y_CONTINUE positive/negative: {labels['positive']}/{labels['negative']}; positive unique outputs: {len(positive_outputs)}.", f"- Best 100%-recall LOO operating point: {best['model']}; continuation={best['continuation_rate']:.3f}; node saving={best['estimated_extra_nodes_saved']:.0f} ({best['extra_node_saving_fraction']:.1%}); false stops={best['false_stop_count']}.", f"- Signal: {status}. This is exploratory because positives <8 and unique outputs <20.", f"- Top descriptive feature candidates: {', '.join(features_ranked[:5])}.", "- 4-fold SHA256 grouping is emitted as an additional exploratory diagnostic; LOO is primary.", "- Pause/resume current implementation: NO / NOT_RUN; see static audit.", f"- Expansion plan: {expansion_rows} rows, estimated {estimated_seconds / 3600:.2f} GPU-hours.", ""]
    (out / "REGRET_BUDGET_ROUTER_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "score"):
        item = sub.add_parser(name); item.add_argument("--d2-run", type=Path, required=True); item.add_argument("--out", type=Path, required=True)
        if name == "score": item.add_argument("--solutions", type=Path, required=True)
    args = parser.parse_args(); {"build": build, "score": score}[args.cmd](args)


if __name__ == "__main__": main()
