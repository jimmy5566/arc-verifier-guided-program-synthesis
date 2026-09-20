"""Score frozen Evaluation30 candidate pools after the target-blind barrier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN = "EVAL30_SEARCH_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
S0_FROZEN = "EVAL30_REUSED_AUG8_GREEDY_CANDIDATES_FROZEN"
CONDITIONS = ("s0", "beam2", "dfs_small", "dfs_medium")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _candidate_pool_exact(record: dict[str, Any], target: Any) -> list[int]:
    return [index for index, candidate in enumerate(record.get("candidates", ())) if candidate.get("prediction") == target]


def _validate_pool(artifact: dict[str, Any], *, task_ids: list[str], manifest: dict[str, Any], condition: str, search_config: dict[str, Any]) -> None:
    expected = set(task_ids)
    if condition == "s0":
        status_ok = artifact.get("status") == S0_FROZEN
    else:
        status_ok = artifact.get("status") == FROZEN and artifact.get("condition") == condition and artifact.get("condition_config") == search_config["conditions"][condition]
    if not status_ok or artifact.get("task_ids_hash") != manifest["task_ids_hash"] or set(artifact.get("records", ())) != expected:
        raise ValueError(f"{condition}: incomplete/non-frozen candidate artifact")
    for task_id in task_ids:
        record = artifact["records"][task_id]
        if record.get("status") != "SUCCESS" or not isinstance(record.get("candidates"), list) or not record["candidates"]:
            raise ValueError(f"{condition}/{task_id}: invalid candidate record")


def _metrics(artifact: dict[str, Any], solved: dict[str, bool]) -> dict[str, Any]:
    records = artifact["records"]
    recovered = [task_id for task_id, value in solved.items() if value]
    gpu_seconds = float(artifact.get("gpu_seconds", sum(float(value.get("elapsed_seconds", 0.0)) for value in records.values())))
    return {
        "any_of_k": len(recovered),
        "recovered_task_ids": recovered,
        "mean_unique_candidates_per_task": sum(int(value.get("unique_candidate_count", len(value.get("candidates", ())))) for value in records.values()) / len(records),
        "mean_invalid_candidates_per_task": sum(int(value.get("invalid_candidate_count", 0)) for value in records.values()) / len(records),
        "runtime_seconds": float(artifact.get("runtime_seconds", 0.0)),
        "gpu_seconds": gpu_seconds,
        "gpu_seconds_per_new_recovery": None if not recovered else gpu_seconds / len(recovered),
        "unique_candidate_count": sum(int(value.get("unique_candidate_count", len(value.get("candidates", ())))) for value in records.values()),
        "invalid_candidate_count": sum(int(value.get("invalid_candidate_count", 0)) for value in records.values()),
    }


def _diagnosis(values: dict[str, dict[str, Any]]) -> str:
    best = max(values[name]["any_of_k"] for name in ("beam2", "dfs_small", "dfs_medium"))
    if best >= 3:
        return "SEARCH_WORKS"
    if best:
        return "WEAK_SEARCH_GAIN"
    return "SEARCH_FAILURE"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "search_config", "s0", "beam2", "dfs_small", "dfs_medium", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    if any((output_dir / item).exists() for item in ("per_task_results.csv", "EVAL30_SEARCH_REPORT.json", "EVAL30_SEARCH_REPORT.md")):
        raise FileExistsError("refusing to overwrite a completed Evaluation30 score")
    manifest, search_config = _read(args.manifest), _read(args.search_config)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL30_COHORT_FROZEN_BEFORE_NEW_SEARCH" or len(task_ids) != 30 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid frozen Evaluation30 manifest")
    if _sha256(args.challenge_path) != manifest.get("source_challenge_sha256"):
        raise ValueError("evaluation challenge changed after cohort freeze")
    artifacts = {name: _read(getattr(args, name)) for name in CONDITIONS}
    for name, artifact in artifacts.items():
        _validate_pool(artifact, task_ids=task_ids, manifest=manifest, condition=name, search_config=search_config)
    # The freeze gate above is intentionally complete before this first target access.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("evaluation solutions omit a frozen task")
    pool_hits: dict[str, dict[str, bool]] = {name: {} for name in CONDITIONS}
    candidate_indices: dict[str, dict[str, list[int]]] = {name: {} for name in CONDITIONS}
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        target = solutions[task_id]
        row: dict[str, Any] = {"task_id": task_id}
        for name in CONDITIONS:
            record = artifacts[name]["records"][task_id]
            exact = _candidate_pool_exact(record, target)
            hit = bool(exact)
            pool_hits[name][task_id] = hit
            candidate_indices[name][task_id] = exact
            row[f"{name}_any_of_k"] = hit
            row[f"{name}_correct_candidate_indices"] = json.dumps(exact)
            row[f"{name}_unique_candidates"] = int(record.get("unique_candidate_count", len(record["candidates"])))
            row[f"{name}_invalid_candidates"] = int(record.get("invalid_candidate_count", 0))
            row[f"{name}_elapsed_seconds"] = float(record.get("elapsed_seconds", 0.0))
        rows.append(row)
    if any(pool_hits["s0"].values()):
        raise AssertionError("S0 is not the declared baseline-pool-miss cohort")
    metrics = {name: _metrics(artifacts[name], pool_hits[name]) for name in CONDITIONS}
    for name in ("beam2", "dfs_small", "dfs_medium"):
        metrics[name]["new_recoveries_vs_s0"] = metrics[name]["any_of_k"]
    report = {
        "experiment_id": "ARC2_EVAL30_CANDIDATE_SEARCH_DIAGNOSTIC",
        "status": "COMPLETE_SCORED_AFTER_ALL_CONDITION_CANDIDATES_FROZEN",
        "protocol": "S0 reused from frozen Evaluation60 Aug8. S1/S2/S3 candidate pools were complete and immutable before this scorer opened evaluation solutions.",
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "source_challenge_sha256": manifest["source_challenge_sha256"],
        "search_config_sha256": _sha256(args.search_config),
        "artifact_sha256": {name: _sha256(getattr(args, name)) for name in CONDITIONS},
        "metrics": metrics,
        "diagnosis": _diagnosis(metrics),
        "interpretation_rule": {"SEARCH_WORKS": "at least 3 new pool recoveries", "WEAK_SEARCH_GAIN": "1-2 new pool recoveries", "SEARCH_FAILURE": "zero new pool recoveries"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output_dir / "EVAL30_SEARCH_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = "\n".join(
        f"| {name} | {values['any_of_k']}/30 | {values['mean_unique_candidates_per_task']:.2f} | {values['mean_invalid_candidates_per_task']:.2f} | {values['runtime_seconds']:.3f}s | {values['gpu_seconds_per_new_recovery'] if values['gpu_seconds_per_new_recovery'] is not None else 'n/a'} |"
        for name, values in metrics.items()
    )
    (output_dir / "EVAL30_SEARCH_REPORT.md").write_text(
        "# Evaluation30 candidate-search diagnostic\n\n"
        f"- Cohort hash: `{manifest['task_ids_hash']}`\n"
        "- Target access occurred only after all S0/S1/S2/S3 candidate files passed the freeze gate.\n"
        "\n| Condition | Any-of-K | Mean unique candidates | Mean invalid candidates | Wall runtime | GPU seconds/new recovery |\n"
        "|---|---:|---:|---:|---:|---:|\n" + table + "\n\n"
        f"- Beam2 new recoveries: {metrics['beam2']['new_recoveries_vs_s0']}\n"
        f"- DFS small new recoveries: {metrics['dfs_small']['new_recoveries_vs_s0']}\n"
        f"- DFS medium new recoveries: {metrics['dfs_medium']['new_recoveries_vs_s0']}\n"
        f"- Diagnosis: **{report['diagnosis']}**\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "S0_ANYK": f"{metrics['s0']['any_of_k']}/30", "S1_ANYK": f"{metrics['beam2']['any_of_k']}/30",
        "S2_ANYK": f"{metrics['dfs_small']['any_of_k']}/30", "S3_ANYK": f"{metrics['dfs_medium']['any_of_k']}/30",
        "BEAM2_NEW_RECOVERIES": metrics["beam2"]["new_recoveries_vs_s0"],
        "DFS_SMALL_NEW_RECOVERIES": metrics["dfs_small"]["new_recoveries_vs_s0"],
        "DFS_MEDIUM_NEW_RECOVERIES": metrics["dfs_medium"]["new_recoveries_vs_s0"],
        "DIAGNOSIS": report["diagnosis"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
