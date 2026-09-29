"""D2-0: align frozen Gold-prefix survival with post-freeze D1 smoke pools.

This is a deliberately nonblind, CPU-only diagnostic.  It reads the already
frozen G1 Gold-prefix token traces and D1's post-freeze Gold-score report.  It
does not import a model, generate a token, or write into any candidate pool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


MAX_SCORE = -math.log(0.2)
POLICIES = (
    "D1_V5_CAPPED_CONTROL",
    "TOPK_LOCAL_k=2",
    "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400",
    "CUMULATIVE_REGRET_r=4.00",
)
SUCCESS_CELL = "20270e3b:o0:d24:identity"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def cell_key(payload: dict[str, Any]) -> str:
    return f"{payload['output_id']}:d{int(payload['depth'])}:{payload['view']}"


def ranked_legal(row: dict[str, Any]) -> list[tuple[int, float]]:
    values = [(int(token), float(logprob)) for token, logprob in json.loads(row["legal_arc_token_logprobs_json"]).items()]
    if len(values) != 12 or not all(math.isfinite(logprob) for _token, logprob in values):
        raise RuntimeError("invalid frozen legal-token vector")
    return sorted(values, key=lambda pair: (-pair[1], pair[0]))


def survives(policy: str, trace: list[dict[str, Any]]) -> bool:
    """Replay only the local Gold-prefix predicates used by D0/D1."""
    prefix_nll = 0.0
    prefix_regret = 0.0
    alive = True
    for row in sorted(trace, key=lambda value: int(value["token_position"])):
        ranked = ranked_legal(row)
        gold = int(row["gold_token_id"])
        by_token = dict(ranked)
        if gold not in by_token:
            raise RuntimeError("Gold token absent from the frozen legal ARC vocabulary")
        if policy == "D1_V5_CAPPED_CONTROL":
            kept = {token for token, logprob in ranked if prefix_nll - logprob < MAX_SCORE}
            if not kept:
                kept = {ranked[0][0]}
        elif policy == "TOPK_LOCAL_k=2":
            kept = {token for token, _logprob in ranked[:2]}
        elif policy == "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400":
            budget = 2.0 + 0.04 * (int(row["token_position"]) + 1)
            kept = {token for token, logprob in ranked if prefix_nll - logprob <= budget + 1e-12}
            if not kept:
                kept = {ranked[0][0]}
        elif policy == "CUMULATIVE_REGRET_r=4.00":
            top_logprob = ranked[0][1]
            kept = {token for token, logprob in ranked if prefix_regret + top_logprob - logprob <= 4.0 + 1e-12}
            if not kept:
                kept = {ranked[0][0]}
        else:  # pragma: no cover - caller owns policy inventory
            raise ValueError(policy)
        alive = alive and gold in kept
        prefix_regret += ranked[0][1] - by_token[gold]
        prefix_nll = float(row["cumulative_gold_nll"])
    return alive


def load_g1_cells(root: Path) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "cells").rglob("*.json")):
        payload = read_json(path)
        key = cell_key(payload)
        if key in cells:
            raise RuntimeError(f"duplicate G1 cell {key}")
        # Keep provenance alongside the parsed trace, but never emit the raw
        # Gold trace in the compact analysis package.
        payload["_source_file_sha256"] = sha256_file(path)
        cells[key] = payload
    if not cells:
        raise RuntimeError("no G1 cell traces found")
    return cells


def choose_cohort(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[str(row["cell_key"])].append(row)
    chosen = [SUCCESS_CELL]
    rationale: dict[str, list[str]] = {SUCCESS_CELL: ["required Regret4 success case"]}
    # Prefer cells that expose survival-but-no-retrieval for several policies;
    # SHA ordering makes all otherwise tied diagnostic choices deterministic.
    candidates = []
    for key, group in by_cell.items():
        if key == SUCCESS_CELL:
            continue
        failure_policies = [str(row["decoder_policy"]) for row in group
                            if as_bool(row["counterfactual_gold_survives"]) and not as_bool(row["actual_gold_hit"])]
        if failure_policies:
            candidates.append((key, failure_policies))
    candidates.sort(key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode("utf-8")).hexdigest(), item[0]))
    coverage: set[str] = set()
    for key, failure_policies in candidates:
        needed = [policy for policy in failure_policies if policy != "D1_V5_CAPPED_CONTROL" and policy not in coverage]
        if needed or (len(chosen) < 6 and failure_policies):
            chosen.append(key)
            coverage.update(failure_policies)
            rationale[key] = ["SURVIVE1_HIT0", *sorted(failure_policies)]
        if len(chosen) >= 8:
            break
    regret_runaway = sorted(
        {str(row["cell_key"]) for row in rows
         if row["decoder_policy"] == "CUMULATIVE_REGRET_r=4.00" and int(float(row["nodes_expanded"] or 0)) >= 3686},
        key=lambda key: (hashlib.sha256(key.encode("utf-8")).hexdigest(), key),
    )
    current_runaway = {key for key in chosen if key in regret_runaway}
    # Make the constraint auditable even where the coverage pass had already
    # selected a runaway before this explicit constraint pass.
    for key in sorted(current_runaway, key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value)):
        rationale.setdefault(key, []).append("Regret4 near-4096-node runaway")
    for key in regret_runaway:
        if len(current_runaway) >= 2 or len(chosen) >= 10:
            break
        if key not in chosen:
            chosen.append(key)
            rationale[key] = ["Regret4 near-4096-node runaway"]
        current_runaway.add(key)
    return chosen, rationale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--d1-gold-cells", type=Path, required=True)
    parser.add_argument("--d1-cost-cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    g1_cells = load_g1_cells(args.g1_root)
    with args.d1_gold_cells.open("r", encoding="utf-8", newline="") as handle:
        gold_rows = list(csv.DictReader(handle))
    with args.d1_cost_cells.open("r", encoding="utf-8", newline="") as handle:
        cost_rows = list(csv.DictReader(handle))
    if len(gold_rows) != 32 or len(cost_rows) != 32:
        raise RuntimeError("D2-0 requires the complete 32-row D1 Smoke8 tables")
    costs = {(row["decoder_policy"], row["cell_key"]): row for row in cost_rows}
    alignment: list[dict[str, Any]] = []
    for gold in gold_rows:
        policy = str(gold["decoder_policy"])
        key = str(gold["cell_key"])
        if policy not in POLICIES or key not in g1_cells:
            raise RuntimeError(f"missing policy/cell alignment evidence: {policy} {key}")
        cost = costs.get((policy, key))
        if cost is None:
            raise RuntimeError(f"missing fixed-cost row: {policy} {key}")
        survived = survives(policy, list(g1_cells[key]["token_trace"]))
        actual = as_bool(gold["gold_hit_any_candidate"])
        classification = f"SURVIVE{int(survived)}_HIT{int(actual)}"
        alignment.append({
            "decoder_policy": policy, "cell_key": key, "output_id": gold["output_id"],
            "counterfactual_gold_survives": survived, "actual_gold_hit": actual,
            "candidate_count": gold["candidate_count"], "nodes_expanded": cost["nodes_expanded"],
            "runtime_seconds": cost["runtime_seconds"], "termination_reason": cost["termination_reason"],
            "alignment_class": classification,
            "d0_source": "G1 frozen incremental-KV Gold-prefix trace replay",
            "d1_candidate_pool_sha256": gold["candidate_pool_sha256"],
        })
    if any(row["alignment_class"] == "SURVIVE0_HIT1" for row in alignment):
        raise RuntimeError("SURVIVE0_HIT1 audit inconsistency: stop before mechanism tracing")
    cohort, rationale = choose_cohort(alignment)
    per_policy = []
    for policy in POLICIES:
        subset = [row for row in alignment if row["decoder_policy"] == policy]
        surviving = sum(as_bool(row["counterfactual_gold_survives"]) for row in subset)
        hits = sum(as_bool(row["actual_gold_hit"]) for row in subset)
        per_policy.append({"decoder_policy": policy, "smoke_cells": len(subset), "surviving_smoke_cells": surviving,
                           "actual_hit_cells": hits, "retrieval_conversion": hits / surviving if surviving else None})
    cohort_payload = {
        "experiment_id": "NONBLIND_MECHANISM_DIAGNOSTIC_D2_V1",
        "cohort_scope": "nonblind development mechanism diagnosis; not an accuracy estimate",
        "selected_cell_keys": cohort,
        "selected_cell_count": len(cohort),
        "selection_rationale": rationale,
        "required_success_cell": SUCCESS_CELL,
        "selection_rule": "success first; maximize surviving-but-not-retrieved policy overlap; then ensure two deterministic Regret4 near-budget cells",
        "source_hashes": {"d1_gold_cells": sha256_file(args.d1_gold_cells), "d1_cost_cells": sha256_file(args.d1_cost_cells)},
    }
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "D2_CELL_ALIGNMENT.csv", alignment)
    write_csv(out / "D2_CELL_ALIGNMENT_SUMMARY.csv", per_policy)
    write_json(out / "D2_MECHANISM_COHORT.json", cohort_payload)
    write_json(out / "D2_INPUT_PROVENANCE.json", {
        "experiment_id": cohort_payload["experiment_id"],
        "input_hashes": cohort_payload["source_hashes"],
        "selected_g1_trace_sha256": {
            key: str(g1_cells[key]["_source_file_sha256"])
            for key in cohort
        },
        "raw_gold_trace_included": False,
    })
    lines = ["# D2 retrieval alignment (CPU-only)", "",
             "D0 Gold-prefix survival was recomputed from frozen G1 incremental-KV traces and joined only to post-freeze D1 Smoke8 pools. This is nonblind mechanism evidence, not accuracy validation.", "",
             "| Policy | Surviving smoke cells | Actual-hit cells | Retrieval conversion |", "|---|---:|---:|---:|"]
    for row in per_policy:
        conversion = "NA" if row["retrieval_conversion"] is None else f"{float(row['retrieval_conversion']):.3f}"
        lines.append(f"| {row['decoder_policy']} | {row['surviving_smoke_cells']} | {row['actual_hit_cells']} | {conversion} |")
    lines += ["", f"- Mechanism cohort: `{len(cohort)}` unique cells", "- `SURVIVE0_HIT1` inconsistencies: `0`"]
    (out / "D2_RETRIEVAL_ALIGNMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
