"""CPU-only Base capability scoring and frozen Eval60 alignment.

The module is deliberately fail-closed: no diagnostic source row (and hence no
diagnostic target) is opened until the two-state, 6000-prediction freeze has
been validated.  The waiting path only verifies the immutable Eval60 taxonomy
and writes protocol/schema/readiness receipts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

from foundation_v2_capability_diagnostic_v1.audit import (
    COMPOSITION_CATEGORIES,
    INPUTS,
    engineering_band,
    score_prediction,
)


SCHEMA_VERSION = "BASE_EVAL60_CAPABILITY_ALIGNMENT_V1"
WAITING = "WAITING_FOR_FULL_6000_RAW_FREEZE"
READY = "FULL_6000_RAW_FREEZE_VERIFIED"
TAXONOMY_COMMIT = "60b608b3976058024728b6d894195f1c150a1b55"
EXPECTED_COUNTS = {"capability": 2656, "parameter": 288, "composition": 56}
EXPECTED_TOTAL = 3000
PROFILE_FILES = (
    "BASE_CAPABILITY_PROFILE_V1.json",
    "BASE_CAPABILITY_PROFILE_V1.csv",
    "BASE_CAPABILITY_PROFILE_V1.md",
    "EVAL60_BASE_CAPABILITY_MATRIX_V1.json",
    "EVAL60_BASE_CAPABILITY_MATRIX_V1.csv",
    "EVAL60_BASE_COMPOSITION_ALIGNMENT_V1.csv",
)

ONTOLOGY = {
    "PERCEPTION_SEGMENTATION": ["connected components", "object vs color grouping", "lines", "frames", "holes/enclosures", "repeated motifs", "separators/subgrids"],
    "ATTRIBUTES": ["color", "area", "width", "height", "orientation", "position", "frequency", "uniqueness"],
    "SELECTORS": ["largest", "smallest", "unique", "most frequent", "least frequent", "top-most", "bottom-most", "left-most", "right-most", "nth/order", "relation-conditioned selector"],
    "RELATIONS": ["left/right", "above/below", "same row", "same column", "adjacent/touching", "inside/contains", "overlap/intersection", "nearest/farthest", "aligned", "connected", "same color"],
    "GEOMETRIC_ACTIONS": ["translate", "rotate 90", "rotate 180", "rotate 270", "reflect LR", "reflect UD", "reflect diagonal", "scale"],
    "MOTION": ["fixed displacement", "inferred displacement", "until boundary", "until touching", "until alignment"],
    "COLOR_MASK": ["recolor", "color mapping", "copy reference color", "mask", "union", "intersection", "difference"],
    "CONSTRUCTION": ["copy", "crop", "extract", "pad", "fill", "border", "overlay", "extend line/ray", "complete missing structure", "symmetry completion"],
    "NUMERIC": ["count components", "count colors", "count cells", "compare quantities", "parity", "encode quantity geometrically"],
    "PATTERN_STATE": ["repeat", "periodic pattern", "alternation", "progression", "propagation", "iterative update"],
    "CONTROL": ["if/else on attribute", "if/else on relation", "if/else on count", "branch-balanced conditional tasks"],
}
CAPABILITY_SECTION = {cap: section for section, caps in ONTOLOGY.items() for cap in caps}


class AlignmentError(RuntimeError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, canonical(value) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows and not fields:
        raise AlignmentError(f"CSV_SCHEMA_REQUIRED={path.name}")
    columns = list(fields or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode_csv(row.get(key)) for key in columns})
    os.replace(tmp, path)


def encode_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return canonical(value)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_taxonomy(taxonomy: Path) -> dict[str, Any]:
    ledger = taxonomy / "SHA256SUMS.txt"
    if not ledger.is_file():
        raise AlignmentError("TAXONOMY_LEDGER_MISSING")
    verified: dict[str, str] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = taxonomy / name.strip()
        if not path.is_file():
            raise AlignmentError(f"TAXONOMY_FILE_MISSING={name}")
        actual = sha256_file(path)
        if actual != expected:
            raise AlignmentError(f"TAXONOMY_HASH_MISMATCH={name}")
        verified[name.strip()] = actual
    required = {
        "EVAL60_CAPABILITY_DEMAND_MAP_V1.json",
        "EVAL60_OUTPUT_CAPABILITY_INDEX_V1.csv",
        "EVAL60_FAMILY_MAP_FREEZE.json",
        "EVAL60_HISTORICAL_ORACLE_BY_FAMILY.json",
        "EVAL60_CAPABILITY_ALIGNMENT_PROTOCOL.json",
        "EVAL60_ONTOLOGY_GAPS.json",
    }
    if not required <= verified.keys():
        raise AlignmentError(f"TAXONOMY_LEDGER_INCOMPLETE={sorted(required-verified.keys())}")
    freeze = json.loads((taxonomy / "EVAL60_FAMILY_MAP_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("task_count") != 60 or freeze.get("test_output_count") != 89:
        raise AlignmentError("TAXONOMY_FREEZE_INVALID")
    return {"status": "PASS", "source_commit": TAXONOMY_COMMIT, "verified_sha256": verified, "task_count": 60, "output_count": 89}


def freeze_status(freeze_path: Path | None) -> tuple[bool, dict[str, Any]]:
    if freeze_path is None or not freeze_path.is_file():
        return False, {"freeze_receipt_present": False}
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    conditions = {
        "status_frozen": payload.get("status") == "FROZEN",
        "base_complete": payload.get("BASE_COMPLETE") is True,
        "foundation_v2_complete": payload.get("FOUNDATION_V2_COMPLETE") is True,
        "total_predictions_6000": payload.get("TOTAL_RAW_PREDICTIONS", payload.get("total_predictions")) == 6000,
        "raw_hashes_verified": payload.get("RAW_HASHES_VERIFIED", payload.get("ALL_RAW_HASHES_PASS")) is True,
        "gold_not_started": payload.get("GOLD_SCORING_STARTED", payload.get("GOLD_SCORING_STARTED_AFTER_FREEZE")) is False,
    }
    return all(conditions.values()), {"freeze_receipt_present": True, "freeze_sha256": sha256_file(freeze_path), "conditions": conditions}


def schemas() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARED_NOT_SCORED",
        "row_contracts": {
            "BASE_CAPABILITY_PROFILE_V1": {"rows": 83, "key": ["capability"], "primary_metrics": {"DIRECT_ATOMIC": "exact_accuracy", "MINIMAL_CONTRAST": "contrast_success", "COMPOSITE_ONLY": "exact_accuracy"}},
            "EVAL60_BASE_CAPABILITY_MATRIX_V1": {"rows": 60, "key": ["task_id"], "forbid_task_solvability_average": True},
            "EVAL60_BASE_COMPOSITION_ALIGNMENT_V1": {"rows": "all_3_PLUS_tasks", "key": ["task_id"]},
            "BASE_EVAL60_REQUIRED_CAPABILITY_MATRIX_V1": {"rows": "capabilities_required_by_Eval60", "key": ["capability"]},
            "BASE_EVAL60_FAMILY_MATRIX_V1": {"rows": "primary_families", "key": ["primary_family"]},
            "BASE_EVAL60_ORACLE_ALIGNMENT_V1": {"rows": 89, "key": ["output_id"]},
        },
        "task_statuses": ["REPRESENTATION_GAP_EXPECTED", "REPRESENTATION_RISK_PARTIAL", "PRIMITIVE_SUPPLY_STRONG", "COMPOSITION_EVIDENCE_NEEDED", "ONTOLOGY_GAP", "INSUFFICIENT_EVIDENCE"],
        "engineering_bands": {"WEAK": "score < 0.40", "PARTIAL": "0.40 <= score < 0.75", "STRONG": "0.75 <= score < 0.95", "SATURATED": "score >= 0.95"},
    }


def gold_preference_protocol() -> dict[str, Any]:
    return {
        "schema_version": "EVAL60_BASE_GOLD_PATH_PREFERENCE_PROTOCOL_V1",
        "status": "PREREGISTERED_NOT_RUN",
        "Gold_preference_analysis_started": False,
        "model_states": ["Base", "Foundation-V2"],
        "future_metrics": ["Gold next-token Top1", "Gold next-token Top2", "Gold next-token Top5", "Gold-vs-best-wrong logit margin", "teacher-forced Gold NLL", "cumulative Gold NLL", "cumulative regret", "first local-rank degradation depth", "Gold-prefix rank survival curve"],
        "claim_boundary": "Capability diagnostics do not measure preference for the Eval60 Gold trajectory.",
        "authorization_required": True,
    }


def write_waiting(out: Path, taxonomy_audit: dict[str, Any], freeze_audit: dict[str, Any], *, base_count: int | None, foundation_count: int | None, freeze_path: Path | None) -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": WAITING,
        "full_6000_raw_freeze_established": False,
        "freeze_path": freeze_path.as_posix() if freeze_path else None,
        "freeze_audit": freeze_audit,
        "observed_counts": {"base": base_count, "foundation_v2": foundation_count, "required_each": 3000},
        "taxonomy_audit": taxonomy_audit,
        "diagnostic_gold_accessed": False,
        "Base_scoring_started": False,
        "Foundation_V2_results_used": False,
        "Eval60_taxonomy_modified": False,
        "Gold_preference_analysis_started": False,
        "next_action": "RERUN_THIS_CPU_ONLY_PIPELINE_AFTER_FULL_6000_RAW_FREEZE",
    }
    atomic_json(out / "BASE_EVAL60_ALIGNMENT_READINESS.json", receipt)
    atomic_json(out / "ALIGNMENT_OUTPUT_SCHEMAS_V1.json", schemas())
    atomic_json(out / "EVAL60_BASE_GOLD_PATH_PREFERENCE_PROTOCOL_V1.json", gold_preference_protocol())
    write_ledger(out)
    return receipt


def band(score: float) -> str:
    return engineering_band(score)


def borderline(score: float, width: float) -> bool:
    return any(abs(score - threshold) <= width + 1e-12 for threshold in (0.40, 0.75, 0.95))


def metric(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AlignmentError("EMPTY_METRIC_GROUP")
    return {
        "sample_count": len(rows),
        "exact_correct": sum(bool(row["exact_grid"]) for row in rows),
        "exact_accuracy": sum(bool(row["exact_grid"]) for row in rows) / len(rows),
        "valid_format_rate": sum(bool(row["valid_format"]) for row in rows) / len(rows),
        "dimension_exact_rate": sum(bool(row["dimension_exact"]) for row in rows) / len(rows),
        "cell_accuracy": sum(float(row["cell_accuracy"]) for row in rows) / len(rows),
    }


def profile_capabilities(scored: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[str(row["capability"])].append(row)
    meta = {row["capability"]: row for row in manifest["capabilities"]}
    result: list[dict[str, Any]] = []
    for capability in sorted(meta):
        rows = grouped[capability]
        kind = meta[capability]["measurement_type"]
        if kind == "MINIMAL_CONTRAST":
            pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                pairs[str(row["pair_id"])].append(row)
            if len(pairs) != 16 or any(len(value) != 2 for value in pairs.values()):
                raise AlignmentError(f"MINIMAL_PAIR_CONTRACT={capability}")
            correct = sum(all(bool(member["exact_grid"]) for member in value) for value in pairs.values())
            score, count, width = correct / 16, 16, 1 / 16
            evidence = "INDEPENDENT_EVIDENCE_" + band(score)
            details = {"pair_count": 16, "contrast_pairs_correct": correct, "contrast_success": score, "individual_task": metric(rows)}
        else:
            details = metric(rows)
            score, count, width = details["exact_accuracy"], len(rows), 1 / len(rows)
            evidence = "COMPOSITE_ONLY_EVIDENCE" if kind == "COMPOSITE_ONLY" else "INDEPENDENT_EVIDENCE_" + band(score)
        result.append({
            "capability": capability,
            "ontology_section": CAPABILITY_SECTION.get(capability, "UNMAPPED"),
            "measurement_type": kind,
            "sample_count_or_pair_count": count,
            "base_primary_score": score,
            "engineering_band": band(score),
            "BORDERLINE": borderline(score, width),
            "borderline_observation_width": width,
            "independent_evidence_status": evidence,
            "parameter_generalization_evidence": [],
            "composition_development_evidence": [],
            "interpretation_scope": meta[capability].get("interpretation_limit"),
            "scoring_rule": meta[capability].get("scoring_rule"),
            "details": details,
        })
    if len(result) != 83:
        raise AlignmentError(f"PROFILE_NOT_83={len(result)}")
    return result


def score_base(input_root: Path, predictions_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = read_jsonl_gz(predictions_path)
    if len(predictions) != EXPECTED_TOTAL or len({row["sample_id"] for row in predictions}) != EXPECTED_TOTAL:
        raise AlignmentError("BASE_RAW_COUNT_OR_ID_FAILURE")
    prediction = {row["sample_id"]: row for row in predictions}
    sources: dict[str, list[dict[str, Any]]] = {}
    scored: dict[str, list[dict[str, Any]]] = {}
    parse_counts: Counter[str] = Counter()
    for cohort, (name, count, expected_sha) in INPUTS.items():
        path = input_root / name
        if sha256_file(path) != expected_sha:
            raise AlignmentError(f"DIAGNOSTIC_SOURCE_HASH_MISMATCH={name}")
        rows = read_jsonl_gz(path)
        if len(rows) != count:
            raise AlignmentError(f"DIAGNOSTIC_SOURCE_COUNT_MISMATCH={name}")
        sources[cohort] = rows
        output: list[dict[str, Any]] = []
        for row in rows:
            pred = prediction.get(row["sample_id"])
            if pred is None or pred.get("model_state") != "base":
                raise AlignmentError(f"BASE_PREDICTION_IDENTITY_FAILURE={row['sample_id']}")
            result = score_prediction(row["sample_id"], pred["generated_token_ids"], row["task"]["test"][0]["output"])
            enriched = {**{key: value for key, value in row.items() if key not in {"task", "scene_evidence", "mechanical_witness"}}, **result}
            output.append(enriched)
            parse_counts[result["parse_failure_class"]] += 1
        scored[cohort] = output
    return scored["capability"], scored["parameter"], scored["composition"], dict(sorted(parse_counts.items()))


def axis_and_composition(parameter: Sequence[dict[str, Any]], composition: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    axes = []
    for axis in sorted({row["axis"] for row in parameter}):
        base = metric([row for row in parameter if row["axis"] == axis and row["domain"] == "base"])
        extra = metric([row for row in parameter if row["axis"] == axis and row["domain"] == "diagnostic"])
        axes.append({"axis": axis, "base_domain": base, "extrapolation_domain": extra, "generalization_gap": extra["exact_accuracy"] - base["exact_accuracy"]})
    comps = []
    for category in COMPOSITION_CATEGORIES:
        values = [row for row in composition if row["category"] == category]
        summary = metric(values)
        comps.append({"category": category, **summary, "engineering_band": band(summary["exact_accuracy"])})
    return axes, comps


def attach_supporting_evidence(profile: Sequence[dict[str, Any]], axes: Sequence[dict[str, Any]], comps: Sequence[dict[str, Any]]) -> None:
    """Attach only conservative, ontology-section-level supporting evidence."""
    axis_lookup = {row["axis"]: row for row in axes}
    comp_lookup = {row["category"]: row for row in comps}
    section_axes = {
        "ATTRIBUTES": {"background_color", "color_permutation", "object_position", "object_size", "orientation"},
        "SELECTORS": {"distractor_count", "object_count", "object_position", "object_size"},
        "RELATIONS": {"object_position", "distractor_count"},
        "GEOMETRIC_ACTIONS": {"orientation", "object_position", "grid_size"},
        "MOTION": {"displacement_magnitude", "object_position", "grid_size"},
        "COLOR_MASK": {"background_color", "color_permutation"},
        "CONSTRUCTION": {"grid_size", "object_size", "object_count"},
        "NUMERIC": {"object_count", "grid_size"},
        "PATTERN_STATE": {"grid_size", "object_count"},
        "PERCEPTION_SEGMENTATION": {"distractor_count", "object_count", "object_size"},
    }
    section_compositions = {
        "GEOMETRIC_ACTIONS": {"GEOMETRY_TO_GEOMETRY"},
        "MOTION": {"GEOMETRY_TO_GEOMETRY"},
        "SELECTORS": {"SELECTOR_TO_ACTION"},
        "RELATIONS": {"RELATION_TO_SELECTOR_ACTION"},
        "NUMERIC": {"COUNTING_TO_CONSTRUCTION"},
        "COLOR_MASK": {"MASK_SET_TO_CONSTRUCTION"},
        "PATTERN_STATE": {"STATE_PROGRESSION_TO_ACTION"},
        "CONTROL": {"CONDITIONAL_TO_ACTION"},
    }
    for row in profile:
        section = row["ontology_section"]
        row["parameter_generalization_evidence"] = [axis_lookup[name] for name in sorted(section_axes.get(section, set())) if name in axis_lookup]
        row["composition_development_evidence"] = [comp_lookup[name] for name in sorted(section_compositions.get(section, set())) if name in comp_lookup]


def composition_matches(task: dict[str, Any]) -> list[str]:
    families = {task["primary_family"], *task.get("secondary_families", [])}
    matches = []
    if families & {"GEOMETRIC_TRANSFORM", "MOTION"}: matches.append("GEOMETRY_TO_GEOMETRY")
    if "OBJECT_SELECTION" in families: matches.append("SELECTOR_TO_ACTION")
    if task.get("requires_relation") and task.get("requires_selector"): matches.append("RELATION_TO_SELECTOR_ACTION")
    if "COUNTING_NUMERIC" in families and "CONSTRUCTION" in families: matches.append("COUNTING_TO_CONSTRUCTION")
    if "MASK_SET" in families and "CONSTRUCTION" in families: matches.append("MASK_SET_TO_CONSTRUCTION")
    if task.get("requires_state_or_progression"): matches.append("STATE_PROGRESSION_TO_ACTION")
    if task.get("requires_conditional_control"): matches.append("CONDITIONAL_TO_ACTION")
    return list(dict.fromkeys(matches))


def task_status(task: dict[str, Any], caps: Sequence[dict[str, Any]], matching_compositions: Sequence[dict[str, Any]]) -> str:
    if task["ontology_coverage"] == "OUTSIDE_CURRENT_ONTOLOGY":
        return "ONTOLOGY_GAP"
    independent = [row for row in caps if row and row["measurement_type"] != "COMPOSITE_ONLY"]
    missing = [row for row in caps if not row or row["measurement_type"] == "COMPOSITE_ONLY"]
    if not independent or len(missing) > len(caps) / 2:
        return "INSUFFICIENT_EVIDENCE"
    bands = {row["engineering_band"] for row in independent}
    if "WEAK" in bands:
        return "REPRESENTATION_GAP_EXPECTED"
    if "PARTIAL" in bands:
        return "REPRESENTATION_RISK_PARTIAL"
    if task["estimated_composition_depth"] == "3_PLUS":
        if not matching_compositions or any(row["engineering_band"] not in {"STRONG", "SATURATED"} for row in matching_compositions):
            return "COMPOSITION_EVIDENCE_NEEDED"
    return "PRIMITIVE_SUPPLY_STRONG"


def build_task_matrices(tasks: Sequence[dict[str, Any]], profile: Sequence[dict[str, Any]], compositions: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cap_lookup = {row["capability"]: row for row in profile}
    comp_lookup = {row["category"]: row for row in compositions}
    matrix, comp_rows = [], []
    for task in tasks:
        caps = [cap_lookup.get(name) for name in task["required_capabilities"]]
        mapped = [row for row in caps if row]
        independent = [row for row in mapped if row["measurement_type"] != "COMPOSITE_ONLY"]
        match_names = composition_matches(task) if task["estimated_composition_depth"] == "3_PLUS" else []
        matches = [comp_lookup[name] for name in match_names]
        status = task_status(task, caps, matches)
        scores = [float(row["base_primary_score"]) for row in independent]
        joined = [{"capability": name, "base_score": row["base_primary_score"] if row else None, "base_band": row["engineering_band"] if row else None, "measurement_type": row["measurement_type"] if row else None, "independent_evidence_status": row["independent_evidence_status"] if row else "NOT_DIRECTLY_MEASURED", "borderline": row["BORDERLINE"] if row else None} for name, row in zip(task["required_capabilities"], caps, strict=True)]
        counts = Counter(row["engineering_band"] for row in independent)
        result = {
            **{key: task[key] for key in ("task_id", "test_output_count", "primary_family", "secondary_families", "ontology_coverage", "estimated_composition_depth", "classification_confidence")},
            "required_capabilities": task["required_capabilities"], "required_capability_count": len(caps), "capability_evidence": joined,
            "required_caps_mapped_count": len(mapped), "required_caps_unmapped_count": len(caps)-len(mapped),
            "weak_required_cap_count": counts["WEAK"], "partial_required_cap_count": counts["PARTIAL"], "strong_required_cap_count": counts["STRONG"], "saturated_required_cap_count": counts["SATURATED"],
            "composite_only_required_cap_count": sum(row["measurement_type"] == "COMPOSITE_ONLY" for row in mapped), "not_directly_measured_count": len(caps)-len(independent),
            "min_independent_base_score": min(scores) if scores else None, "median_independent_base_score": statistics.median(scores) if scores else None, "mean_independent_base_score": statistics.fmean(scores) if scores else None,
            "all_independent_required_caps_strong_or_better": bool(independent) and all(row["engineering_band"] in {"STRONG", "SATURATED"} for row in independent),
            "any_required_cap_weak": counts["WEAK"] > 0, "any_required_cap_partial_or_weaker": counts["WEAK"] + counts["PARTIAL"] > 0,
            "base_representation_status": status,
        }
        matrix.append(result)
        if task["estimated_composition_depth"] == "3_PLUS":
            comp_rows.append({"task_id": task["task_id"], "primary_family": task["primary_family"], "required_capabilities": task["required_capabilities"], "composition_depth": "3_PLUS", "mapped_composition_categories": match_names or ["NO_CLEAR_COMPOSITION_MATCH"], "base_composition_scores": [row["exact_accuracy"] for row in matches], "base_composition_bands": [row["engineering_band"] for row in matches], "mapping_confidence": "MEDIUM" if match_names else "LOW"})
    if len(matrix) != 60:
        raise AlignmentError(f"TASK_MATRIX_NOT_60={len(matrix)}")
    return matrix, comp_rows


def freeze_pre_oracle(out: Path) -> dict[str, Any]:
    hashes = {name: sha256_file(out / name) for name in PROFILE_FILES}
    payload = {"status": "FROZEN_BEFORE_HISTORICAL_ORACLE_JOIN", "files": hashes, "classification_mutation_after_join_forbidden": True}
    atomic_json(out / "BASE_MAP_PRE_ORACLE_FREEZE.json", payload)
    return payload


def join_oracle(output_index: Path, oracle_csv: Path, task_matrix: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    with output_index.open(encoding="utf-8", newline="") as handle:
        index = list(csv.DictReader(handle))
    with oracle_csv.open(encoding="utf-8", newline="") as handle:
        oracle = {row["output_id"]: row for row in csv.DictReader(handle)}
    tasks = {row["task_id"]: row for row in task_matrix}
    rows = []
    for item in index:
        outcome = oracle.get(item["output_id"])
        if outcome is None:
            raise AlignmentError(f"ORACLE_OUTPUT_MISSING={item['output_id']}")
        task = tasks[item["task_id"]]
        rows.append({"task_id": item["task_id"], "output_id": item["output_id"], "primary_family": item["primary_family"], "composition_depth": item["estimated_composition_depth"], "ontology_coverage": item["ontology_coverage"], "base_representation_status": task["base_representation_status"], "ORC_UNION": str(outcome["ORC_UNION"]).lower() in {"1", "true"}})
    if len(rows) != 89 or sum(row["ORC_UNION"] for row in rows) != 35:
        raise AlignmentError("ORACLE_35_OF_89_CONTRACT_FAILURE")
    return rows


def required_capability_matrix(tasks: Sequence[dict[str, Any]], profile: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["capability"]: row for row in profile}
    oracle_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle: oracle_by_task[row["task_id"]].append(row)
    result = []
    for capability in sorted({cap for task in tasks for cap in task["required_capabilities"]}):
        selected = [task for task in tasks if capability in task["required_capabilities"]]
        outcomes = [row for task in selected for row in oracle_by_task[task["task_id"]]]
        profile_row = lookup.get(capability)
        result.append({"capability": capability, "Eval60_task_count": len(selected), "Eval60_output_count": len(outcomes), "Base_measurement_type": profile_row["measurement_type"] if profile_row else None, "Base_score": profile_row["base_primary_score"] if profile_row else None, "Base_band": profile_row["engineering_band"] if profile_row else "NOT_INDEPENDENTLY_MEASURED", "borderline": profile_row["BORDERLINE"] if profile_row else None, "historical_ORC_output_count": len(outcomes), "historical_ORC_solved_count": sum(row["ORC_UNION"] for row in outcomes), "historical_ORC_rate": sum(row["ORC_UNION"] for row in outcomes)/len(outcomes)})
    return sorted(result, key=lambda row: (-row["Eval60_output_count"], row["Base_score"] if row["Base_score"] is not None else 2, row["capability"]))


def family_matrix(tasks: Sequence[dict[str, Any]], task_matrix: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]], compositions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    source = {row["task_id"]: row for row in tasks}; tm = {row["task_id"]: row for row in task_matrix}
    result = []
    for family in sorted({row["primary_family"] for row in tasks}):
        selected = [row for row in tasks if row["primary_family"] == family]
        outcomes = [row for row in oracle if row["primary_family"] == family]
        cap_rows = [cap for row in selected for cap in tm[row["task_id"]]["capability_evidence"]]
        scores = [float(row["base_score"]) for row in cap_rows if row["base_score"] is not None and row["measurement_type"] != "COMPOSITE_ONLY"]
        band_counts = Counter(row["base_band"] if row["measurement_type"] != "COMPOSITE_ONLY" else "NOT_INDEPENDENTLY_MEASURED" for row in cap_rows)
        denom = len(cap_rows) or 1
        matches = sorted({name for row in selected for name in composition_matches(source[row["task_id"]])})
        comp_lookup = {row["category"]: row for row in compositions}
        result.append({"primary_family": family, "task_count": len(selected), "output_count": len(outcomes), "historical_ORC_solved": sum(row["ORC_UNION"] for row in outcomes), "historical_ORC_total": len(outcomes), "historical_ORC_rate": sum(row["ORC_UNION"] for row in outcomes)/len(outcomes), "mean_mapped_Base_primitive_score": statistics.fmean(scores) if scores else None, "median_mapped_Base_primitive_score": statistics.median(scores) if scores else None, **{f"fraction_{name}": band_counts[name]/denom for name in ("WEAK", "PARTIAL", "STRONG", "SATURATED", "NOT_INDEPENDENTLY_MEASURED")}, "3_PLUS_composition_fraction": sum(row["estimated_composition_depth"] == "3_PLUS" for row in selected)/len(selected), "Base_composition_development": [{"category": name, "score": comp_lookup[name]["exact_accuracy"], "band": comp_lookup[name]["engineering_band"]} for name in matches], "ontology_gap_count": sum(row["ontology_coverage"] == "OUTSIDE_CURRENT_ONTOLOGY" for row in selected)})
    return result


def quadrant_analysis(task_matrix: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]], compositions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    misses = Counter(row["task_id"] for row in oracle if not row["ORC_UNION"])
    totals = Counter(row["task_id"] for row in oracle)
    buckets = {name: [] for name in ("Q1_BASE_WEAK_OR_PARTIAL__ORC_MISS_HEAVY", "Q2_BASE_STRONG__ORC_HIGH", "Q3_BASE_STRONG__ORC_MISS_HEAVY", "Q4_INSUFFICIENT_OR_ONTOLOGY_GAP")}
    for task in task_matrix:
        miss_heavy = misses[task["task_id"]] / totals[task["task_id"]] >= 0.5
        status = task["base_representation_status"]
        if status in {"ONTOLOGY_GAP", "INSUFFICIENT_EVIDENCE"}:
            bucket = "Q4_INSUFFICIENT_OR_ONTOLOGY_GAP"
        elif status in {"REPRESENTATION_GAP_EXPECTED", "REPRESENTATION_RISK_PARTIAL"}:
            # Keep weak/partial primitive supply out of the explicitly Base-strong
            # quadrant even when the historical oracle happened to solve a task.
            bucket = "Q1_BASE_WEAK_OR_PARTIAL__ORC_MISS_HEAVY"
        else:
            bucket = "Q3_BASE_STRONG__ORC_MISS_HEAVY" if miss_heavy else "Q2_BASE_STRONG__ORC_HIGH"
        buckets[bucket].append(task["task_id"])
    return {
        "status": "COMPLETE_DESCRIPTIVE_ONLY",
        "quadrants": buckets,
        "q1_name_is_descriptive_of_the_target_failure_pattern": True,
        "q1_may_include_ORC_high_tasks_to_preserve_the_four-way_partition": True,
        "causal_conclusion": "NOT_ESTABLISHED",
    }


def special_focus_summary(tasks: Sequence[dict[str, Any]], task_matrix: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]], compositions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tm = {row["task_id"]: row for row in task_matrix}
    comp_lookup = {row["category"]: row for row in compositions}
    selectors = {
        "MULTI_STAGE_COMPOSITION": lambda row: row["primary_family"] == "MULTI_STAGE_COMPOSITION",
        "PATTERN_PROGRESSION": lambda row: row["primary_family"] == "PATTERN_PROGRESSION",
        "COLOR_MAPPING": lambda row: row["primary_family"] == "COLOR_MAPPING",
        "MOTION": lambda row: row["primary_family"] == "MOTION",
        "relation-heavy": lambda row: bool(row.get("requires_relation")),
        "state/progression-heavy": lambda row: bool(row.get("requires_state_or_progression")),
    }
    result: dict[str, Any] = {}
    for label, predicate in selectors.items():
        selected = [row for row in tasks if predicate(row)]
        ids = {row["task_id"] for row in selected}
        outcomes = [row for row in oracle if row["task_id"] in ids]
        bands = Counter(cap["base_band"] if cap["measurement_type"] != "COMPOSITE_ONLY" else "NOT_INDEPENDENTLY_MEASURED" for task in selected for cap in tm[task["task_id"]]["capability_evidence"])
        matches = sorted({name for row in selected for name in composition_matches(row)})
        result[label] = {
            "task_count": len(selected),
            "output_count": len(outcomes),
            "historical_ORC_solved": sum(row["ORC_UNION"] for row in outcomes),
            "historical_ORC_rate": sum(row["ORC_UNION"] for row in outcomes) / len(outcomes) if outcomes else None,
            "Base_required_capability_strength_distribution": dict(bands),
            "Base_composition_evidence": [{"category": name, "score": comp_lookup[name]["exact_accuracy"], "band": comp_lookup[name]["engineering_band"]} for name in matches],
            "task_status_counts": dict(Counter(tm[task_id]["base_representation_status"] for task_id in ids)),
        }
    return result


def miss_interpretation_summary(tasks: Sequence[dict[str, Any]], task_matrix: Sequence[dict[str, Any]], oracle: Sequence[dict[str, Any]], compositions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source = {row["task_id"]: row for row in tasks}
    tm = {row["task_id"]: row for row in task_matrix}
    comp_lookup = {row["category"]: row for row in compositions}
    misses = [row for row in oracle if not row["ORC_UNION"]]

    def primitive_strong(task_id: str) -> bool:
        row = tm[task_id]
        return row["all_independent_required_caps_strong_or_better"] and not row["any_required_cap_partial_or_weaker"]

    def composition_strong(task_id: str) -> bool:
        names = composition_matches(source[task_id])
        return bool(names) and all(comp_lookup[name]["engineering_band"] in {"STRONG", "SATURATED"} for name in names)

    surprising = [row for row in misses if primitive_strong(row["task_id"])]
    expected = [row for row in misses if tm[row["task_id"]]["any_required_cap_partial_or_weaker"]]
    return {
        "historical_ORC_misses_with_strong_Base_primitive_supply": len(surprising),
        "historical_ORC_misses_with_strong_primitive_and_matching_composition_evidence": sum(composition_strong(row["task_id"]) for row in surprising),
        "top_15_surprising_misses": surprising[:15],
        "top_15_expected_misses": expected[:15],
        "ontology_gap_tasks": sorted(row["task_id"] for row in task_matrix if row["base_representation_status"] == "ONTOLOGY_GAP"),
        "interpretation_case_counts": {
            "BASE_REPRESENTATION_GAP_PLAUSIBLE": sum(tm[row["task_id"]]["base_representation_status"] in {"REPRESENTATION_GAP_EXPECTED", "REPRESENTATION_RISK_PARTIAL"} for row in misses),
            "BASE_COMPOSITION_GAP_PLAUSIBLE": sum(primitive_strong(row["task_id"]) and not composition_strong(row["task_id"]) for row in misses),
            "BASE_CAPABILITY_SUPPLY_STRONG__PIPELINE_OR_PREFERENCE_GAP_SUSPECT": sum(primitive_strong(row["task_id"]) and composition_strong(row["task_id"]) for row in misses),
            "DIAGNOSTIC_COVERAGE_INSUFFICIENT": sum(tm[row["task_id"]]["base_representation_status"] in {"INSUFFICIENT_EVIDENCE", "ONTOLOGY_GAP"} for row in misses),
        },
        "causal_conclusion": "NOT_ESTABLISHED",
    }


def write_profile(out: Path, profile: Sequence[dict[str, Any]], axes: Sequence[dict[str, Any]], comps: Sequence[dict[str, Any]]) -> None:
    payload = {"schema_version": "BASE_CAPABILITY_PROFILE_V1", "status": "COMPLETE", "model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "capabilities": list(profile), "parameter_generalization": list(axes), "composition_development": list(comps), "claim_boundary": "COMPOSITE_ONLY evidence cannot establish isolated primitive mastery."}
    atomic_json(out / "BASE_CAPABILITY_PROFILE_V1.json", payload)
    write_csv(out / "BASE_CAPABILITY_PROFILE_V1.csv", list(profile))
    lines = ["# Base Capability Profile V1", "", "COMPOSITE_ONLY evidence is reported separately and cannot establish isolated primitive mastery.", "", "| Capability | Section | Measurement | Score | Band | Independent evidence | Borderline |", "|---|---|---|---:|---|---|---:|"]
    for row in profile: lines.append(f"| {row['capability']} | {row['ontology_section']} | {row['measurement_type']} | {row['base_primary_score']:.4f} | {row['engineering_band']} | {row['independent_evidence_status']} | {row['BORDERLINE']} |")
    atomic_text(out / "BASE_CAPABILITY_PROFILE_V1.md", "\n".join(lines) + "\n")


def write_alignment_report(out: Path, summary: dict[str, Any]) -> None:
    """Render every preregistered summary field without adding causal claims."""
    bands = summary["profile_band_counts"]
    statuses = summary["task_status_counts"]
    lines = [
        "# Base x Eval60 Capability Alignment V1",
        "",
        summary["claim_boundary"],
        "",
        "## Frozen boundaries",
        "",
        f"- Full 6000 raw freeze established before diagnostic Gold access: {summary['full_6000_raw_freeze_established_before_gold']}",
        f"- Base predictions scored: {summary['Base_predictions_scored']}/3000",
        f"- Foundation-V2 results used: {summary['Foundation_V2_results_used']}",
        f"- Eval60 taxonomy modified: {summary['Eval60_taxonomy_modified']}",
        f"- Gold-path preference analysis started: {summary['Gold_preference_analysis_started']}",
        "",
        "## Base capability bands",
        "",
        f"- WEAK: {bands.get('WEAK', 0)}",
        f"- PARTIAL: {bands.get('PARTIAL', 0)}",
        f"- STRONG: {bands.get('STRONG', 0)}",
        f"- SATURATED: {bands.get('SATURATED', 0)}",
        f"- BORDERLINE: {summary['borderline_count']}",
        "",
        "## Eval60 task representation status",
        "",
    ]
    for name in (
        "REPRESENTATION_GAP_EXPECTED",
        "REPRESENTATION_RISK_PARTIAL",
        "PRIMITIVE_SUPPLY_STRONG",
        "COMPOSITION_EVIDENCE_NEEDED",
        "ONTOLOGY_GAP",
        "INSUFFICIENT_EVIDENCE",
    ):
        lines.append(f"- {name}: {statuses.get(name, 0)}")

    lines.extend([
        "",
        "## Highest-frequency Eval60 requirements",
        "",
        "| Capability | Eval60 outputs | Base measurement | Base score | Base band | ORC rate |",
        "|---|---:|---|---:|---|---:|",
    ])
    for row in summary["highest_frequency_required_capabilities"]:
        score = "NOT_MEASURED" if row["Base_score"] is None else f"{row['Base_score']:.4f}"
        lines.append(
            f"| {row['capability']} | {row['Eval60_output_count']} | "
            f"{row['Base_measurement_type'] or 'NOT_DIRECTLY_MEASURED'} | {score} | "
            f"{row['Base_band']} | {row['historical_ORC_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## Special-focus groups",
        "",
        "| Group | Outputs | ORC rate | Required-capability bands | Task statuses | Composition evidence |",
        "|---|---:|---:|---|---|---|",
    ])
    for name, row in summary["special_focus_groups"].items():
        rate = "N/A" if row["historical_ORC_rate"] is None else f"{row['historical_ORC_rate']:.4f}"
        comp = ", ".join(
            f"{item['category']}={item['score']:.4f}({item['band']})"
            for item in row["Base_composition_evidence"]
        ) or "NO_CLEAR_COMPOSITION_MATCH"
        lines.append(
            f"| {name} | {row['output_count']} | {rate} | "
            f"{canonical(row['Base_required_capability_strength_distribution'])} | "
            f"{canonical(row['task_status_counts'])} | {comp} |"
        )

    lines.extend([
        "",
        "## Historical ORC miss alignment",
        "",
        f"- ORC misses with strong Base primitive supply: {summary['historical_ORC_misses_with_strong_Base_primitive_supply']}",
        f"- ORC misses with strong primitives and strong matching composition evidence: {summary['historical_ORC_misses_with_strong_primitive_and_matching_composition_evidence']}",
        f"- Interpretation case counts: {canonical(summary['interpretation_case_counts'])}",
        f"- Ontology-gap tasks: {', '.join(summary['ontology_gap_tasks']) or 'NONE'}",
        "",
        "### Top surprising misses",
        "",
        "| Task | Output | Family | Representation status |",
        "|---|---|---|---|",
    ])
    for row in summary["top_15_surprising_misses"]:
        lines.append(f"| {row['task_id']} | {row['output_id']} | {row['primary_family']} | {row['base_representation_status']} |")

    lines.extend([
        "",
        "### Top expected misses",
        "",
        "| Task | Output | Family | Representation status |",
        "|---|---|---|---|",
    ])
    for row in summary["top_15_expected_misses"]:
        lines.append(f"| {row['task_id']} | {row['output_id']} | {row['primary_family']} | {row['base_representation_status']} |")

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Base diagnostic capability evidence does not measure whether the model prefers the Eval60 Gold trajectory.",
        "Strong primitive evidence therefore weakens only a pure primitive-representation explanation; it does not establish task solvability.",
        "",
    ])
    atomic_text(out / "BASE_EVAL60_ALIGNMENT_REPORT_V1.md", "\n".join(lines))


def write_ledger(out: Path) -> None:
    entries = []
    for path in sorted(out.iterdir(), key=lambda value: value.name):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{sha256_file(path)}  {path.name}")
    atomic_text(out / "SHA256SUMS.txt", "\n".join(entries) + "\n")


def run(*, taxonomy: Path, out: Path, freeze_path: Path | None, input_root: Path | None = None, base_predictions: Path | None = None, oracle_csv: Path | None = None, observed_base_count: int | None = None, observed_foundation_count: int | None = None) -> dict[str, Any]:
    taxonomy_audit = verify_taxonomy(taxonomy)
    ready, freeze_audit = freeze_status(freeze_path)
    if not ready:
        return write_waiting(out, taxonomy_audit, freeze_audit, base_count=observed_base_count, foundation_count=observed_foundation_count, freeze_path=freeze_path)
    if input_root is None or base_predictions is None or oracle_csv is None:
        raise AlignmentError("READY_RUN_REQUIRES_INPUT_ROOT_BASE_PREDICTIONS_ORACLE_CSV")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    base_receipt = freeze["raw_files"]["base"]
    if base_receipt.get("rows") != 3000 or sha256_file(base_predictions) != base_receipt.get("sha256"):
        raise AlignmentError("BASE_PREDICTION_FREEZE_MISMATCH")
    # This is the first operation that can load diagnostic Gold.
    cap_scored, parameter_scored, composition_scored, parse_counts = score_base(input_root, base_predictions)
    manifest = json.loads((input_root / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.json").read_text(encoding="utf-8"))
    profile = profile_capabilities(cap_scored, manifest)
    axes, compositions = axis_and_composition(parameter_scored, composition_scored)
    attach_supporting_evidence(profile, axes, compositions)
    tasks_payload = json.loads((taxonomy / "EVAL60_CAPABILITY_DEMAND_MAP_V1.json").read_text(encoding="utf-8"))
    tasks = tasks_payload["tasks"]
    task_matrix, comp_matrix = build_task_matrices(tasks, profile, compositions)
    out.mkdir(parents=True, exist_ok=True)
    write_profile(out, profile, axes, compositions)
    atomic_json(out / "EVAL60_BASE_CAPABILITY_MATRIX_V1.json", {"schema_version": "EVAL60_BASE_CAPABILITY_MATRIX_V1", "status": "COMPLETE", "tasks": task_matrix, "task_solvability_score_created": False})
    write_csv(out / "EVAL60_BASE_CAPABILITY_MATRIX_V1.csv", task_matrix)
    write_csv(out / "EVAL60_BASE_COMPOSITION_ALIGNMENT_V1.csv", comp_matrix)
    pre_oracle = freeze_pre_oracle(out)
    oracle = join_oracle(taxonomy / "EVAL60_OUTPUT_CAPABILITY_INDEX_V1.csv", oracle_csv, task_matrix)
    atomic_json(out / "BASE_EVAL60_ORACLE_ALIGNMENT_V1.json", {"schema_version": "BASE_EVAL60_ORACLE_ALIGNMENT_V1", "status": "COMPLETE", "pre_oracle_freeze": pre_oracle, "outputs": oracle})
    write_csv(out / "BASE_EVAL60_ORACLE_ALIGNMENT_V1.csv", oracle)
    required = required_capability_matrix(tasks, profile, oracle)
    families = family_matrix(tasks, task_matrix, oracle, compositions)
    write_csv(out / "BASE_EVAL60_REQUIRED_CAPABILITY_MATRIX_V1.csv", required)
    write_csv(out / "BASE_EVAL60_FAMILY_MATRIX_V1.csv", families)
    quadrants = quadrant_analysis(task_matrix, oracle, compositions)
    atomic_json(out / "BASE_EVAL60_QUADRANT_ANALYSIS_V1.json", quadrants)
    status_counts = Counter(row["base_representation_status"] for row in task_matrix)
    profile_counts = Counter(row["engineering_band"] for row in profile)
    focus = special_focus_summary(tasks, task_matrix, oracle, compositions)
    miss_summary = miss_interpretation_summary(tasks, task_matrix, oracle, compositions)
    summary = {"schema_version": "BASE_EVAL60_ALIGNMENT_SUMMARY_V1", "status": "COMPLETE", "full_6000_raw_freeze_established_before_gold": True, "Base_predictions_scored": 3000, "Foundation_V2_results_used": False, "Eval60_taxonomy_modified": False, "Gold_preference_analysis_started": False, "diagnostic_gold_accessed_after_freeze": True, "profile_band_counts": dict(profile_counts), "borderline_count": sum(row["BORDERLINE"] for row in profile), "task_status_counts": dict(status_counts), "parse_failure_counts": parse_counts, "historical_ORC": {"solved": 35, "total": 89}, "highest_frequency_required_capabilities": required[:15], "special_focus_groups": focus, **miss_summary, "claim_boundary": "Strong primitive evidence weakens a pure primitive-representation explanation but does not imply Eval60 task solvability or Gold-path preference."}
    atomic_json(out / "BASE_EVAL60_ALIGNMENT_SUMMARY_V1.json", summary)
    write_alignment_report(out, summary)
    atomic_json(out / "BASE_EVAL60_ALIGNMENT_READINESS.json", {"schema_version": SCHEMA_VERSION, "status": READY, "full_6000_raw_freeze_established": True, "freeze_audit": freeze_audit, "taxonomy_audit": taxonomy_audit, "diagnostic_gold_accessed_after_freeze": True})
    atomic_json(out / "ALIGNMENT_OUTPUT_SCHEMAS_V1.json", schemas())
    atomic_json(out / "EVAL60_BASE_GOLD_PATH_PREFERENCE_PROTOCOL_V1.json", gold_preference_protocol())
    write_ledger(out)
    return summary
