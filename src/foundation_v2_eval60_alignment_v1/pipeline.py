"""CPU-only alignment of frozen Foundation-V2 diagnostics with Eval60 demand.

No model or raw prediction is loaded here.  The pipeline consumes only the
hash-verified, post-freeze diagnostic summaries and the already-frozen Base x
Eval60 analysis.  Foundation classifications are frozen before historical ORC
outcomes are joined.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from base_eval60_capability_alignment_v1.pipeline import (
    CAPABILITY_SECTION,
    AlignmentError as BaseAlignmentError,
    atomic_json,
    atomic_text,
    band,
    borderline,
    build_task_matrices,
    canonical,
    composition_matches,
    sha256_file,
    verify_taxonomy,
    write_csv,
)


SCHEMA_VERSION = "FOUNDATION_V2_EVAL60_CAPABILITY_ALIGNMENT_V1"
SOURCE_COMMIT = "ada422e7be9f77267ef28dcef449f568f982b909"
EXPECTED_PROFILE_COUNT = 83
EXPECTED_TASK_COUNT = 60
EXPECTED_OUTPUT_COUNT = 89

PRE_ORACLE_FILES = (
    "FOUNDATION_V2_CAPABILITY_PROFILE_V1.json",
    "FOUNDATION_V2_CAPABILITY_PROFILE_V1.csv",
    "FOUNDATION_V2_CAPABILITY_PROFILE_V1.md",
    "EVAL60_FOUNDATION_V2_CAPABILITY_MATRIX_V1.json",
    "EVAL60_FOUNDATION_V2_CAPABILITY_MATRIX_V1.csv",
    "EVAL60_FOUNDATION_V2_COMPOSITION_ALIGNMENT_V1.csv",
    "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1.json",
    "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1.csv",
)


class AlignmentError(BaseAlignmentError):
    pass


def verify_ledger(directory: Path) -> dict[str, str]:
    ledger = directory / "SHA256SUMS.txt"
    if not ledger.is_file():
        raise AlignmentError(f"SHA_LEDGER_MISSING={directory}")
    verified: dict[str, str] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        path = directory / name
        if not path.is_file():
            raise AlignmentError(f"LEDGER_FILE_MISSING={path}")
        actual = sha256_file(path)
        if actual != expected:
            raise AlignmentError(f"LEDGER_HASH_MISMATCH={path}")
        verified[name] = actual
    return verified


def source_audit(base_dir: Path, foundation_dir: Path, taxonomy_dir: Path) -> dict[str, Any]:
    base_hashes = verify_ledger(base_dir)
    foundation_hashes = verify_ledger(foundation_dir)
    taxonomy = verify_taxonomy(taxonomy_dir)
    freeze = json.loads((foundation_dir / "RAW_PREDICTION_FREEZE.json").read_text(encoding="utf-8"))
    data_access = json.loads((foundation_dir / "DATA_ACCESS_AUDIT.json").read_text(encoding="utf-8"))
    no_training = json.loads((foundation_dir / "NO_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    conditions = {
        "full_6000_frozen": freeze.get("status") == "FROZEN" and freeze.get("TOTAL_RAW_PREDICTIONS") == 6000,
        "base_complete": freeze.get("BASE_COMPLETE") is True,
        "foundation_v2_complete": freeze.get("FOUNDATION_V2_COMPLETE") is True,
        "raw_hashes_verified": freeze.get("RAW_HASHES_VERIFIED") is True and freeze.get("ALL_RAW_HASHES_PASS") is True,
        "gold_started_after_freeze": freeze.get("GOLD_SCORING_STARTED_AFTER_FREEZE") is True,
        "source_eval60_gold_not_accessed": data_access.get("EVAL60_GOLD_ACCESSED") is False,
        "source_forbidden_access_audit": data_access.get("status") == "PASS",
        "no_training": no_training.get("status") == "PASS" and no_training.get("optimizer_steps") == 0,
        "base_analysis_complete": "BASE_EVAL60_ALIGNMENT_SUMMARY_V1.json" in base_hashes,
        "foundation_profile_complete": "FOUNDATION_V2_CAPABILITY_PROFILE.json" in foundation_hashes,
    }
    if not all(conditions.values()):
        raise AlignmentError(f"SOURCE_AUDIT_FAIL={conditions}")
    return {
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "conditions": conditions,
        "base_artifact_count": len(base_hashes),
        "foundation_artifact_count": len(foundation_hashes),
        "taxonomy_audit": taxonomy,
        "base_analysis_ledger_sha256": sha256_file(base_dir / "SHA256SUMS.txt"),
        "foundation_diagnostic_ledger_sha256": sha256_file(foundation_dir / "SHA256SUMS.txt"),
        "raw_prediction_freeze_sha256": sha256_file(foundation_dir / "RAW_PREDICTION_FREEZE.json"),
    }


def normalize_foundation_profile(source: Mapping[str, Any], base_profile: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(source.get("capabilities", []))
    if source.get("status") != "COMPLETE" or len(rows) != EXPECTED_PROFILE_COUNT:
        raise AlignmentError("FOUNDATION_PROFILE_NOT_COMPLETE_83")
    base_lookup = {row["capability"]: row for row in base_profile}
    source_lookup = {row["capability"]: row for row in rows}
    if set(base_lookup) != set(source_lookup):
        raise AlignmentError("BASE_FOUNDATION_CAPABILITY_IDENTITY_MISMATCH")
    result: list[dict[str, Any]] = []
    for capability in sorted(source_lookup):
        row = source_lookup[capability]
        base_row = base_lookup[capability]
        if float(row["base_primary_score"]) != float(base_row["base_primary_score"]):
            raise AlignmentError(f"BASE_SCORE_PARITY_FAIL={capability}")
        measurement = str(row["measurement_type"])
        score = float(row["foundation_v2_primary_score"])
        width = float(row["borderline_observation_width"])
        evidence = "COMPOSITE_ONLY_EVIDENCE" if measurement == "COMPOSITE_ONLY" else f"INDEPENDENT_EVIDENCE_{band(score)}"
        result.append({
            "capability": capability,
            "ontology_section": CAPABILITY_SECTION.get(capability, "UNMAPPED"),
            "measurement_type": measurement,
            "sample_count_or_pair_count": int(row["foundation_v2"].get("total", base_row["sample_count_or_pair_count"])),
            "foundation_v2_primary_score": score,
            "engineering_band": band(score),
            "BORDERLINE": borderline(score, width),
            "borderline_observation_width": width,
            "independent_evidence_status": evidence,
            "base_primary_score": float(row["base_primary_score"]),
            "delta_vs_base": score - float(row["base_primary_score"]),
            "details": row["foundation_v2"],
            "claim_boundary": "COMPOSITE_ONLY evidence does not establish isolated primitive mastery.",
        })
    return result


def normalize_compositions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "COMPLETE":
        raise AlignmentError("FOUNDATION_COMPOSITION_RESULTS_INCOMPLETE")
    result = []
    for row in payload["categories"]:
        score = float(row["foundation_v2"]["exact_accuracy"])
        result.append({
            "category": row["category"],
            "exact_accuracy": score,
            "engineering_band": band(score),
            "base_exact_accuracy": float(row["base"]["exact_accuracy"]),
            "delta_vs_base": float(row["delta"]),
            **{key: value for key, value in row["foundation_v2"].items() if key != "exact_accuracy"},
        })
    if len(result) != 7:
        raise AlignmentError(f"FOUNDATION_COMPOSITION_COUNT={len(result)}")
    return result


def _as_base_internal(profile: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "base_primary_score": row["foundation_v2_primary_score"]} for row in profile]


def _rename_foundation_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_rename_foundation_keys(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        name = key.replace("base_", "foundation_v2_").replace("Base_", "Foundation_V2_")
        result[name] = _rename_foundation_keys(item)
    return result


def combine_task_matrices(base_tasks: Sequence[dict[str, Any]], foundation_tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    base_lookup = {row["task_id"]: row for row in base_tasks}
    foundation_lookup = {row["task_id"]: row for row in foundation_tasks}
    if set(base_lookup) != set(foundation_lookup) or len(base_lookup) != EXPECTED_TASK_COUNT:
        raise AlignmentError("TASK_MATRIX_IDENTITY_FAILURE")
    result = []
    for task_id in sorted(base_lookup):
        base = base_lookup[task_id]
        foundation = foundation_lookup[task_id]
        evidence = []
        for left, right in zip(base["capability_evidence"], foundation["capability_evidence"], strict=True):
            if left["capability"] != right["capability"]:
                raise AlignmentError(f"TASK_CAPABILITY_ORDER_MISMATCH={task_id}")
            bscore, fscore = left["base_score"], right["foundation_v2_score"]
            evidence.append({
                "capability": left["capability"],
                "measurement_type": left["measurement_type"],
                "Base_score": bscore,
                "Base_band": left["base_band"],
                "Foundation_V2_score": fscore,
                "Foundation_V2_band": right["foundation_v2_band"],
                "Foundation_V2_minus_Base": None if bscore is None or fscore is None else float(fscore) - float(bscore),
                "independent_measurement": left["measurement_type"] not in {None, "COMPOSITE_ONLY"},
            })
        result.append({
            **{key: base[key] for key in ("task_id", "test_output_count", "primary_family", "secondary_families", "ontology_coverage", "estimated_composition_depth", "classification_confidence", "required_capabilities")},
            "Base_representation_status": base["base_representation_status"],
            "Foundation_V2_representation_status": foundation["foundation_v2_representation_status"],
            "status_transition": f"{base['base_representation_status']}->{foundation['foundation_v2_representation_status']}",
            "Base_min_independent_score": base["min_independent_base_score"],
            "Foundation_V2_min_independent_score": foundation["min_independent_foundation_v2_score"],
            "Base_mean_independent_score": base["mean_independent_base_score"],
            "Foundation_V2_mean_independent_score": foundation["mean_independent_foundation_v2_score"],
            "mean_independent_score_delta": None if base["mean_independent_base_score"] is None or foundation["mean_independent_foundation_v2_score"] is None else float(foundation["mean_independent_foundation_v2_score"]) - float(base["mean_independent_base_score"]),
            "capability_evidence": evidence,
            "task_solvability_score_created": False,
        })
    return result


def freeze_pre_oracle(out: Path) -> dict[str, Any]:
    files = {name: sha256_file(out / name) for name in PRE_ORACLE_FILES}
    payload = {
        "status": "FROZEN_BEFORE_HISTORICAL_ORACLE_JOIN",
        "files": files,
        "classification_mutation_after_join_forbidden": True,
        "historical_oracle_accessed_before_this_freeze": False,
    }
    atomic_json(out / "FOUNDATION_V2_MAP_PRE_ORACLE_FREEZE.json", payload)
    return payload


def join_oracle(base_oracle: Sequence[dict[str, Any]], combined_tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = {row["task_id"]: row for row in combined_tasks}
    result = []
    for row in base_oracle:
        task = tasks[row["task_id"]]
        result.append({
            "task_id": row["task_id"],
            "output_id": row["output_id"],
            "primary_family": row["primary_family"],
            "composition_depth": row["composition_depth"],
            "ontology_coverage": row["ontology_coverage"],
            "Base_representation_status": task["Base_representation_status"],
            "Foundation_V2_representation_status": task["Foundation_V2_representation_status"],
            "status_transition": task["status_transition"],
            "historical_ORC_UNION": bool(row["ORC_UNION"]),
        })
    if len(result) != EXPECTED_OUTPUT_COUNT or sum(row["historical_ORC_UNION"] for row in result) != 35:
        raise AlignmentError("ORACLE_35_OF_89_CONTRACT_FAILURE")
    return result


def combined_capability_matrix(tasks: Sequence[dict[str, Any]], base_profile: Sequence[dict[str, Any]], foundation_profile: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    base_lookup = {row["capability"]: row for row in base_profile}
    foundation_lookup = {row["capability"]: row for row in foundation_profile}
    oracle_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle:
        oracle_by_task[row["task_id"]].append(row)
    result = []
    for capability in sorted({cap for task in tasks for cap in task["required_capabilities"]}):
        selected = [task for task in tasks if capability in task["required_capabilities"]]
        outcomes = [row for task in selected for row in oracle_by_task[task["task_id"]]]
        base = base_lookup.get(capability)
        foundation = foundation_lookup.get(capability)
        bscore = None if base is None else float(base["base_primary_score"])
        fscore = None if foundation is None else float(foundation["foundation_v2_primary_score"])
        delta = None if bscore is None or fscore is None else fscore - bscore
        trend = "NOT_MEASURED" if delta is None else ("IMPROVED" if delta > 1e-12 else "REGRESSED" if delta < -1e-12 else "UNCHANGED")
        result.append({
            "capability": capability,
            "Eval60_task_count": len(selected),
            "Eval60_output_count": len(outcomes),
            "measurement_type": None if base is None else base["measurement_type"],
            "independently_measured": base is not None and base["measurement_type"] != "COMPOSITE_ONLY",
            "Base_score": bscore,
            "Base_band": "NOT_INDEPENDENTLY_MEASURED" if base is None else base["engineering_band"],
            "Base_borderline": None if base is None else base["BORDERLINE"],
            "Foundation_V2_score": fscore,
            "Foundation_V2_band": "NOT_INDEPENDENTLY_MEASURED" if foundation is None else foundation["engineering_band"],
            "Foundation_V2_borderline": None if foundation is None else foundation["BORDERLINE"],
            "Foundation_V2_minus_Base": delta,
            "score_trend": trend,
            "historical_ORC_output_count": len(outcomes),
            "historical_ORC_solved_count": sum(row["historical_ORC_UNION"] for row in outcomes),
            "historical_ORC_rate": sum(row["historical_ORC_UNION"] for row in outcomes) / len(outcomes),
        })
    return sorted(result, key=lambda row: (-row["Eval60_output_count"], row["Foundation_V2_score"] if row["Foundation_V2_score"] is not None else 2, row["capability"]))


def combined_family_matrix(tasks: Sequence[dict[str, Any]], combined_tasks: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = {row["task_id"]: row for row in combined_tasks}
    result = []
    for family in sorted({row["primary_family"] for row in tasks}):
        selected = [row for row in tasks if row["primary_family"] == family]
        ids = {row["task_id"] for row in selected}
        outcomes = [row for row in oracle if row["task_id"] in ids]
        evidence = [cap for task_id in ids for cap in combined[task_id]["capability_evidence"] if cap["independent_measurement"]]
        base_scores = [float(row["Base_score"]) for row in evidence if row["Base_score"] is not None]
        foundation_scores = [float(row["Foundation_V2_score"]) for row in evidence if row["Foundation_V2_score"] is not None]
        result.append({
            "primary_family": family,
            "task_count": len(selected),
            "output_count": len(outcomes),
            "historical_ORC_solved": sum(row["historical_ORC_UNION"] for row in outcomes),
            "historical_ORC_total": len(outcomes),
            "historical_ORC_rate": sum(row["historical_ORC_UNION"] for row in outcomes) / len(outcomes),
            "Base_mean_independent_required_score": statistics.fmean(base_scores) if base_scores else None,
            "Foundation_V2_mean_independent_required_score": statistics.fmean(foundation_scores) if foundation_scores else None,
            "Foundation_V2_minus_Base_mean": None if not base_scores or not foundation_scores else statistics.fmean(foundation_scores) - statistics.fmean(base_scores),
            "Base_task_status_counts": dict(Counter(combined[row["task_id"]]["Base_representation_status"] for row in selected)),
            "Foundation_V2_task_status_counts": dict(Counter(combined[row["task_id"]]["Foundation_V2_representation_status"] for row in selected)),
            "ontology_gap_count": sum(row["ontology_coverage"] == "OUTSIDE_CURRENT_ONTOLOGY" for row in selected),
        })
    return result


def write_profile(out: Path, profile: Sequence[dict[str, Any]], source_audit_payload: Mapping[str, Any]) -> None:
    atomic_json(out / "FOUNDATION_V2_CAPABILITY_PROFILE_V1.json", {
        "schema_version": "FOUNDATION_V2_CAPABILITY_PROFILE_V1",
        "status": "COMPLETE_FROM_HASH_VERIFIED_FROZEN_DIAGNOSTIC",
        "source_audit": source_audit_payload,
        "capabilities": list(profile),
        "claim_boundary": "COMPOSITE_ONLY evidence cannot establish isolated primitive mastery.",
    })
    write_csv(out / "FOUNDATION_V2_CAPABILITY_PROFILE_V1.csv", list(profile))
    lines = [
        "# Foundation-V2 Capability Profile V1", "",
        "Built CPU-only from the hash-verified frozen 3000-sample diagnostic. COMPOSITE_ONLY evidence remains non-independent.", "",
        "| Capability | Measurement | Base | Foundation-V2 | Delta | Band | Independent evidence |", "|---|---|---:|---:|---:|---|---|",
    ]
    for row in profile:
        lines.append(f"| {row['capability']} | {row['measurement_type']} | {row['base_primary_score']:.4f} | {row['foundation_v2_primary_score']:.4f} | {row['delta_vs_base']:+.4f} | {row['engineering_band']} | {row['independent_evidence_status']} |")
    atomic_text(out / "FOUNDATION_V2_CAPABILITY_PROFILE_V1.md", "\n".join(lines) + "\n")


def write_report(out: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Base / Foundation-V2 / Eval60 Capability Alignment V1", "",
        summary["claim_boundary"], "", "## Source and safety", "",
        f"- Source audit: {summary['source_audit_status']}",
        f"- Base diagnostic rows: {summary['Base_predictions_scored']}/3000",
        f"- Foundation-V2 diagnostic rows: {summary['Foundation_V2_predictions_scored']}/3000",
        f"- Eval60 taxonomy modified: {summary['Eval60_taxonomy_modified']}",
        f"- New GPU/model inference: {summary['new_GPU_or_model_inference']}",
        f"- Gold-path preference analysis started: {summary['Gold_preference_analysis_started']}",
        "", "## Capability bands", "",
        "| Model | WEAK | PARTIAL | STRONG | SATURATED | BORDERLINE |", "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("Base", "Foundation_V2"):
        counts = summary["profile_band_counts"][model]
        lines.append(f"| {model} | {counts.get('WEAK',0)} | {counts.get('PARTIAL',0)} | {counts.get('STRONG',0)} | {counts.get('SATURATED',0)} | {summary['borderline_counts'][model]} |")
    lines.extend(["", "## Eval60 task representation status", "", "| Status | Base | Foundation-V2 |", "|---|---:|---:|"])
    statuses = ("REPRESENTATION_GAP_EXPECTED", "REPRESENTATION_RISK_PARTIAL", "PRIMITIVE_SUPPLY_STRONG", "COMPOSITION_EVIDENCE_NEEDED", "ONTOLOGY_GAP", "INSUFFICIENT_EVIDENCE")
    for status in statuses:
        lines.append(f"| {status} | {summary['task_status_counts']['Base'].get(status,0)} | {summary['task_status_counts']['Foundation_V2'].get(status,0)} |")
    lines.extend(["", "## Highest-frequency Eval60 requirements", "", "| Capability | Eval60 outputs | Measurement | Base | Foundation-V2 | Delta | ORC rate |", "|---|---:|---|---:|---:|---:|---:|"])
    for row in summary["highest_frequency_required_capabilities"]:
        b = "N/A" if row["Base_score"] is None else f"{row['Base_score']:.4f} ({row['Base_band']})"
        f = "N/A" if row["Foundation_V2_score"] is None else f"{row['Foundation_V2_score']:.4f} ({row['Foundation_V2_band']})"
        d = "N/A" if row["Foundation_V2_minus_Base"] is None else f"{row['Foundation_V2_minus_Base']:+.4f}"
        lines.append(f"| {row['capability']} | {row['Eval60_output_count']} | {row['measurement_type'] or 'NOT_MEASURED'} | {b} | {f} | {d} | {row['historical_ORC_rate']:.4f} |")
    lines.extend(["", "## Largest Eval60-demand capability changes", "", "| Capability | Eval60 outputs | Base | Foundation-V2 | Delta |", "|---|---:|---:|---:|---:|"])
    for row in summary["largest_absolute_capability_changes"]:
        lines.append(f"| {row['capability']} | {row['Eval60_output_count']} | {row['Base_score']:.4f} | {row['Foundation_V2_score']:.4f} | {row['Foundation_V2_minus_Base']:+.4f} |")
    lines.extend(["", "## Interpretation boundary", "", "No task-solvability average was created. Capability supply is descriptive and does not measure Gold-trajectory preference or establish Eval60 task solvability.", ""])
    atomic_text(out / "BASE_FOUNDATION_V2_EVAL60_ALIGNMENT_REPORT_V1.md", "\n".join(lines))


def write_ledger(out: Path) -> None:
    entries = []
    for path in sorted(out.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{sha256_file(path)}  {path.name}")
    atomic_text(out / "SHA256SUMS.txt", "\n".join(entries) + "\n")


def run(*, base_dir: Path, foundation_dir: Path, taxonomy_dir: Path, out: Path) -> dict[str, Any]:
    audit = source_audit(base_dir, foundation_dir, taxonomy_dir)
    base_profile_payload = json.loads((base_dir / "BASE_CAPABILITY_PROFILE_V1.json").read_text(encoding="utf-8"))
    base_profile = base_profile_payload["capabilities"]
    foundation_source = json.loads((foundation_dir / "FOUNDATION_V2_CAPABILITY_PROFILE.json").read_text(encoding="utf-8"))
    foundation_profile = normalize_foundation_profile(foundation_source, base_profile)
    compositions = normalize_compositions(json.loads((foundation_dir / "COMPOSITION_DEVELOPMENT_RESULTS.json").read_text(encoding="utf-8")))
    tasks = json.loads((taxonomy_dir / "EVAL60_CAPABILITY_DEMAND_MAP_V1.json").read_text(encoding="utf-8"))["tasks"]
    base_tasks = json.loads((base_dir / "EVAL60_BASE_CAPABILITY_MATRIX_V1.json").read_text(encoding="utf-8"))["tasks"]
    internal_tasks, internal_compositions = build_task_matrices(tasks, _as_base_internal(foundation_profile), compositions)
    foundation_tasks = _rename_foundation_keys(internal_tasks)
    foundation_compositions = _rename_foundation_keys(internal_compositions)
    combined_tasks = combine_task_matrices(base_tasks, foundation_tasks)

    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "SOURCE_AUDIT.json", audit)
    write_profile(out, foundation_profile, audit)
    atomic_json(out / "EVAL60_FOUNDATION_V2_CAPABILITY_MATRIX_V1.json", {"schema_version": "EVAL60_FOUNDATION_V2_CAPABILITY_MATRIX_V1", "status": "COMPLETE", "tasks": foundation_tasks, "task_solvability_score_created": False})
    write_csv(out / "EVAL60_FOUNDATION_V2_CAPABILITY_MATRIX_V1.csv", foundation_tasks)
    write_csv(out / "EVAL60_FOUNDATION_V2_COMPOSITION_ALIGNMENT_V1.csv", foundation_compositions)
    atomic_json(out / "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1.json", {"schema_version": "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1", "status": "COMPLETE", "tasks": combined_tasks, "task_solvability_score_created": False})
    write_csv(out / "BASE_FOUNDATION_V2_EVAL60_TASK_MATRIX_PRE_ORACLE_V1.csv", combined_tasks)
    pre_oracle = freeze_pre_oracle(out)

    base_oracle = json.loads((base_dir / "BASE_EVAL60_ORACLE_ALIGNMENT_V1.json").read_text(encoding="utf-8"))["outputs"]
    oracle = join_oracle(base_oracle, combined_tasks)
    atomic_json(out / "BASE_FOUNDATION_V2_EVAL60_ORACLE_ALIGNMENT_V1.json", {"schema_version": "BASE_FOUNDATION_V2_EVAL60_ORACLE_ALIGNMENT_V1", "status": "COMPLETE", "pre_oracle_freeze": pre_oracle, "outputs": oracle})
    write_csv(out / "BASE_FOUNDATION_V2_EVAL60_ORACLE_ALIGNMENT_V1.csv", oracle)
    capabilities = combined_capability_matrix(tasks, base_profile, foundation_profile, oracle)
    atomic_json(out / "BASE_FOUNDATION_V2_EVAL60_CAPABILITY_MATRIX_V1.json", {"schema_version": "BASE_FOUNDATION_V2_EVAL60_CAPABILITY_MATRIX_V1", "status": "COMPLETE", "rows": capabilities, "task_solvability_score_created": False})
    write_csv(out / "BASE_FOUNDATION_V2_EVAL60_CAPABILITY_MATRIX_V1.csv", capabilities)
    families = combined_family_matrix(tasks, combined_tasks, oracle)
    write_csv(out / "BASE_FOUNDATION_V2_EVAL60_FAMILY_MATRIX_V1.csv", families)

    base_bands = Counter(row["engineering_band"] for row in base_profile)
    foundation_bands = Counter(row["engineering_band"] for row in foundation_profile)
    base_statuses = Counter(row["base_representation_status"] for row in base_tasks)
    foundation_statuses = Counter(row["foundation_v2_representation_status"] for row in foundation_tasks)
    transitions = Counter(row["status_transition"] for row in combined_tasks)
    measured = [row for row in capabilities if row["Foundation_V2_minus_Base"] is not None]
    changes = sorted(measured, key=lambda row: (-abs(row["Foundation_V2_minus_Base"]), -row["Eval60_output_count"], row["capability"]))[:15]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "source_audit_status": audit["status"],
        "Base_predictions_scored": 3000,
        "Foundation_V2_predictions_scored": 3000,
        "full_6000_raw_freeze_established_before_diagnostic_gold": True,
        "new_GPU_or_model_inference": False,
        "Eval60_taxonomy_modified": False,
        "Gold_preference_analysis_started": False,
        "profile_band_counts": {"Base": dict(base_bands), "Foundation_V2": dict(foundation_bands)},
        "borderline_counts": {"Base": sum(row["BORDERLINE"] for row in base_profile), "Foundation_V2": sum(row["BORDERLINE"] for row in foundation_profile)},
        "task_status_counts": {"Base": dict(base_statuses), "Foundation_V2": dict(foundation_statuses)},
        "task_status_transitions": dict(sorted(transitions.items())),
        "capability_score_trends": dict(Counter(row["score_trend"] for row in capabilities)),
        "highest_frequency_required_capabilities": capabilities[:15],
        "largest_absolute_capability_changes": changes,
        "historical_ORC": {"solved": 35, "total": 89},
        "capability_matrix_rows": len(capabilities),
        "task_matrix_rows": len(combined_tasks),
        "family_matrix_rows": len(families),
        "claim_boundary": "Base and Foundation-V2 diagnostic evidence measures capability supply, not Eval60 task solvability or Gold-path preference.",
    }
    atomic_json(out / "BASE_FOUNDATION_V2_EVAL60_ALIGNMENT_SUMMARY_V1.json", summary)
    write_report(out, summary)
    write_ledger(out)
    return summary
