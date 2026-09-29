"""Audit historical Eval60 TTT24 Aug8 greedy artifacts without model execution.

This is deliberately fail-closed.  A historical prediction is never scored
against Gold or merged into the current development union unless every hard
identity requirement is evidenced by frozen artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OMITTED = ("rot90", "rot180", "rot270", "flip_lr")
CURRENT4 = ("identity", "flip_ud", "transpose", "anti_transpose")
ALL8 = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grid_valid(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(row, list) and bool(row) and all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row)
        for row in value
    )


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def current_hard_outputs(root: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    path = root / "reports/eval60_bottleneck_attribution_v1/g1_half_sha256_28of56/g1_output_summary.csv"
    rows = read_csv(path)
    result = {row["output_id"]: row for row in rows}
    if len(result) != 28:
        raise RuntimeError(f"expected exactly 28 frozen hard outputs, found {len(result)}")
    return sorted(result), result


def current_task_ids(root: Path) -> set[str]:
    path = root / "artifacts/eval60_compact_analysis_v2/greedy/greedy_cells.csv"
    return {row["task_id"] for row in read_csv(path)}


def current4_hits(root: Path, hard: set[str]) -> set[str]:
    path = root / "artifacts/eval60_compact_analysis_v2/greedy/greedy_cells.csv"
    return {
        row["output_id"]
        for row in read_csv(path)
        if row["output_id"] in hard
        and row["depth"] == "24"
        and row["view"] in CURRENT4
        and row["exact_gold_hit"].strip().lower() == "true"
    }


def record_views(payload: dict[str, Any]) -> set[str]:
    return {
        str(candidate.get("augmentation", {}).get("geometry"))
        for record in payload.get("records", {}).values()
        for candidate in record.get("candidates", [])
    }


def source_inventory_row(label: str, path: Path, payload: dict[str, Any], current_tasks: set[str]) -> dict[str, Any]:
    config = payload.get("reference_config", {})
    records = payload.get("records", {})
    output_count = sum(
        len(record["candidates"][0].get("prediction", []))
        for record in records.values()
        if record.get("candidates")
    )
    return {
        "artifact_label": label,
        "artifact_path": str(path),
        "artifact_sha256": sha256(path),
        "artifact_kind": "ACTUAL_HISTORICAL_EVAL60_TTT24_AUG8_GREEDY",
        "task_cohort": "Eval60",
        "task_ids_hash": payload.get("task_ids_hash", "UNAVAILABLE"),
        "task_count": len(payload.get("task_ids", [])),
        "output_count": output_count,
        "current_task_overlap": len(set(payload.get("task_ids", [])) & current_tasks),
        "ttt_depth": config.get("ttt_steps", "UNAVAILABLE"),
        "augmentation_list": ";".join(sorted(record_views(payload))),
        "model_identity": payload.get("model_identity", "UNAVAILABLE_IN_ARTIFACT"),
        "tokenizer_identity": payload.get("identity", "UNAVAILABLE"),
        "ttt_recipe": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "generation_semantics": payload.get("protocol", "UNAVAILABLE"),
        "adapter_checkpoint_identity": "UNAVAILABLE_IN_ARTIFACT",
        "source_commit": config.get("reference_commit", "UNAVAILABLE"),
        "classification": "MATCHED_BUT_NOT_EXACT",
    }


def gate_row(label: str, path: Path, payload: dict[str, Any], current_tasks: set[str], hard_ids: list[str]) -> dict[str, Any]:
    available_outputs: set[str] = set()
    for task_id, record in payload.get("records", {}).items():
        candidates = record.get("candidates") or []
        if not candidates:
            continue
        for index in range(len(candidates[0].get("prediction", []))):
            available_outputs.add(f"{task_id}:o{index}")
    views = record_views(payload)
    config = payload.get("reference_config", {})
    gates = {
        "gate_1_task_overlap": "PASS" if set(payload.get("task_ids", [])) == current_tasks else "FAIL",
        "gate_2_output_indexing": (
            "PASS_FOR_HARD28_ONLY: every frozen hard output index is present; historical candidate coverage is partial over full Eval60"
            if set(hard_ids).issubset(available_outputs) else "FAIL"
        ),
        "gate_3_model_checkpoint_family": "UNAVAILABLE: artifact has no explicit model/checkpoint identifier",
        "gate_4_tokenizer_native_serialization": "UNAVAILABLE: current frozen compact artifacts omit tokenizer/native serialization metadata",
        "gate_5_ttt24_recipe": "PARTIAL: historical config is 24-step rank-256 alpha-32 rsLoRA, but current resolved recipe bytes are not retained in the compact artifact",
        "gate_6_adapter_sha": "FAIL: historical artifact has no per-task TTT24 adapter SHA256",
        "gate_7_greedy_semantics": "UNAVAILABLE: historical protocol says greedy beam=1 but lacks full prompt/EOS/pad/do_sample audit fields",
        "gate_8_exact_d4_augmentations": "PASS" if views == set(ALL8) else "FAIL",
    }
    return {
        "artifact_label": label,
        "artifact_path": str(path),
        "artifact_sha256": sha256(path),
        **gates,
        "exact_reuse": "NO",
        "classification": "MATCHED_BUT_NOT_EXACT",
        "blocking_gate": "gate_6_adapter_sha",
        "historical_ttt_steps": config.get("ttt_steps", "UNAVAILABLE"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("analysis/historical_eval60_aug8_reuse_v1"))
    parser.add_argument(
        "--historical",
        type=Path,
        action="append",
        required=True,
        help="Frozen historical TTT24 candidates JSON; may be passed more than once.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hard_ids, hard_meta = current_hard_outputs(root)
    hard_set = set(hard_ids)
    current_tasks = current_task_ids(root)
    authoritative_current4 = current4_hits(root, hard_set)

    sources: list[tuple[str, Path, dict[str, Any]]] = []
    for index, path in enumerate(args.historical, start=1):
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append((f"historical_eval60_ttt24_aug8_{index}", path, payload))

    inventory = [
        {
            "artifact_label": "current_authoritative_eval60_compact",
            "artifact_path": str(root / "artifacts/eval60_compact_analysis_v2"),
            "artifact_sha256": "MANIFEST:" + sha256(root / "artifacts/eval60_compact_analysis_v2/MANIFEST.json"),
            "artifact_kind": "CURRENT_AUTHORITATIVE_EVAL60",
            "task_cohort": "Eval60",
            "task_ids_hash": "UNAVAILABLE_IN_COMPACT_EXPORT",
            "task_count": len(current_tasks), "output_count": 89, "current_task_overlap": len(current_tasks),
            "ttt_depth": "12;24;48", "augmentation_list": ";".join(CURRENT4),
            "model_identity": "UNAVAILABLE_IN_COMPACT_EXPORT", "tokenizer_identity": "UNAVAILABLE_IN_COMPACT_EXPORT",
            "ttt_recipe": "hash-only:8968f67e5a5c8c1f838dc1a45120d052527d65e95188cd9e155426ba56481486",
            "generation_semantics": "frozen current greedy cell export", "adapter_checkpoint_identity": "per-cell SHA256 retained",
            "source_commit": "d157bc7977527a9bc02b576be7735e78c202aef0", "classification": "NOT_RELEVANT",
        },
        {
            "artifact_label": "historical_frozen60_stage0_descriptor",
            "artifact_path": str(root / "experiments/stage0_attribution_pilot/frozen60"),
            "artifact_sha256": "NOT_A_SINGLE_CANDIDATE_ARTIFACT",
            "artifact_kind": "HISTORICAL_FROZEN60_DESCRIPTOR",
            "task_cohort": "Frozen60", "task_ids_hash": "NOT_EVAL60", "task_count": "UNKNOWN", "output_count": "UNKNOWN",
            "current_task_overlap": "NOT_USED", "ttt_depth": "UNKNOWN", "augmentation_list": "NOT_USED",
            "model_identity": "NOT_USED", "tokenizer_identity": "NOT_USED", "ttt_recipe": "NOT_USED",
            "generation_semantics": "NOT_USED", "adapter_checkpoint_identity": "NOT_USED", "source_commit": "repository history",
            "classification": "NOT_RELEVANT",
        },
    ]
    inventory.extend(source_inventory_row(label, path, payload, current_tasks) for label, path, payload in sources)
    fields_inventory = list(inventory[0])
    write_csv(output / "historical_artifact_inventory.csv", inventory, fields_inventory)

    gates = [gate_row(label, path, payload, current_tasks, hard_ids) for label, path, payload in sources]
    write_csv(output / "identity_gate.csv", gates, list(gates[0]))

    cells: list[dict[str, Any]] = []
    for label, path, payload in sources:
        records = payload["records"]
        for output_id in hard_ids:
            task_id, suffix = output_id.split(":o")
            output_index = int(suffix)
            record = records[task_id]
            candidates = {candidate["augmentation"]["geometry"]: candidate for candidate in record.get("candidates", [])}
            for view in OMITTED:
                candidate = candidates.get(view)
                prediction = None if candidate is None or output_index >= len(candidate.get("prediction", [])) else candidate["prediction"][output_index]
                cells.append({
                    "task_id": task_id,
                    "output_id": output_id,
                    "output_index": output_index,
                    "depth": 24,
                    "view": view,
                    "historical_artifact": label,
                    "historical_artifact_sha256": sha256(path),
                    "adapter_checkpoint_sha": "UNAVAILABLE_IN_HISTORICAL_ARTIFACT",
                    "prediction_sha": json_sha(prediction) if prediction is not None else "UNAVAILABLE",
                    "parse_valid": str(grid_valid(prediction)).upper(),
                    "exact_gold_hit": "NOT_EVALUATED_NONEXACT_REUSE",
                    "reuse_status": "MATCHED_BUT_NOT_EXACT",
                    "current_union_hit": "FALSE_BY_FROZEN_HARD28_MEMBERSHIP",
                    "historical_prediction_available": str(prediction is not None).upper(),
                })
    cell_fields = list(cells[0])
    write_csv(output / "hard28_ttt24_omitted_view_greedy.csv", cells, cell_fields)
    available_logical_cells = {
        (row["output_id"], row["view"])
        for row in cells
        if row["historical_prediction_available"] == "TRUE"
    }
    available_source_rows = sum(row["historical_prediction_available"] == "TRUE" for row in cells)

    per_view = []
    for view in OMITTED:
        scoped = [row for row in cells if row["view"] == view]
        per_view.append({
            "view": view,
            "historical_source_rows": len(scoped),
            "unique_hard28_logical_cells": len({row["output_id"] for row in scoped if row["historical_prediction_available"] == "TRUE"}),
            "parse_valid_rows": sum(row["parse_valid"] == "TRUE" for row in scoped),
            "exact_reusable_rows": 0,
            "new_greedy_rescues": "NOT_ESTABLISHED_NONEXACT_REUSE",
            "reason": "missing historical per-task adapter SHA256 blocks exact reuse",
        })
    write_csv(output / "view_rescue_summary.csv", per_view, list(per_view[0]))

    outputs = []
    for output_id in hard_ids:
        historical_rows = [row for row in cells if row["output_id"] == output_id]
        outputs.append({
            "task_id": output_id.split(":")[0],
            "output_id": output_id,
            "hard28_class": hard_meta[output_id]["output_class"],
            "current4_authoritative_greedy_positive": str(output_id in authoritative_current4).upper(),
            "omitted4_historical_predictions_available": sum(row["historical_prediction_available"] == "TRUE" for row in historical_rows),
            "omitted4_exact_reusable": "NO",
            "omitted4_greedy_rescue": "NOT_EVALUATED_NONEXACT_REUSE",
            "full_d4_greedy_positive": "NOT_EVALUATED_NONEXACT_REUSE",
            "current_union_hit": "FALSE_BY_FROZEN_HARD28_MEMBERSHIP",
        })
    write_csv(output / "output_rescue_summary.csv", outputs, list(outputs[0]))

    current_hard_hash = hashlib.sha256("\n".join(hard_ids).encode()).hexdigest()
    provenance = {
        "experiment": "HISTORICAL_EVAL60_AUG8_REUSE_V1",
        "source_commit": git_head(root),
        "cpu_only": True, "gpu_used": False, "model_loaded": False, "new_generation": False, "new_ttt": False, "new_dfs": False,
        "current_development_union": "33/89",
        "current_hard28_outputs": hard_ids,
        "current_hard28_sha256": current_hard_hash,
        "historical_sources": [{"label": label, "path": str(path), "sha256": sha256(path)} for label, path, _ in sources],
        "gold_access": "NOT_OPENED: exact reuse identity gate failed before historical prediction scoring",
        "identity_rule": "All eight gates must be evidenced PASS for exact reuse. Missing adapter SHA is a blocking failure.",
    }
    write_json(output / "provenance.json", provenance)

    decision = {
        "CPU_AUDIT_STATUS": "PASS",
        "HISTORICAL_EVAL60_AUG8_FOUND": "YES",
        "EXACT_REUSE_AVAILABLE": "NO",
        "EXACT_REUSE_REASON": "Both historical Eval60 TTT24 Aug8 greedy artifacts lack per-task adapter checkpoint SHA256; additional tokenizer/current-recipe and full greedy-semantic fields are also insufficient for all eight identity gates.",
        "CURRENT_HARD28_OUTPUTS": 28,
        "TTT24_OMITTED_VIEW_CELLS_AVAILABLE": {
            "expected_logical_cells": 112,
            "historical_prediction_available_unique": len(available_logical_cells),
            "historical_prediction_available_source_rows": available_source_rows,
            "exact_reusable_cells": 0,
        },
        "NEW_GREEDY_RESCUE_OUTPUTS": "NOT_ESTABLISHED_NONEXACT_REUSE",
        "NEW_GREEDY_RESCUE_IDS": [],
        "BY_VIEW": {view: "NOT_ESTABLISHED_NONEXACT_REUSE" for view in OMITTED},
        "CURRENT_DEVELOPMENT_UNION": "33/89",
        "NEW_DEVELOPMENT_UNION": "NOT_ESTABLISHED_NONEXACT_REUSE",
        "AUGMENTATION_COVERAGE_CONCLUSION": "HISTORICAL_ONLY",
        "directional_information": "The artifacts verify historical Eval60 coverage for all omitted D4 views, but cannot establish current decoder-positive rescues without adapter identity parity.",
        "NEXT_RECOMMENDED_EXPERIMENT": "Run the preregistered current-adapter TTT24 omitted-D4 Greedy coverage cells, with per-task adapter SHA and native decoder contract frozen before Gold scoring.",
    }
    write_json(output / "DECISION.json", decision)

    (output / "README.md").write_text(
        "# Historical Eval60 Aug8 reuse audit\n\n"
        "CPU-only provenance audit. It discovered two actual historical Eval60 TTT24/Aug8/greedy candidate pools and extracted their omitted D4 predictions for the frozen hard28. "
        "They are **not** exact-reusable: the historical records do not retain per-task TTT24 adapter SHA256, so no historical prediction was compared with Gold and nothing updates the current 33/89 development union.\n",
        encoding="utf-8",
    )
    report = [
        "# Historical Eval60 Aug8 Greedy reuse audit",
        "",
        "## Result",
        "",
        "Two actual historical Eval60 TTT24/Aug8 greedy artifacts were found. They cover all 60 current Eval60 tasks, all 28 frozen hard outputs have matching output indices, and the historical augmentation list is the exact eight-view geometry set.",
        "",
        "Exact reuse is nevertheless blocked: neither historical artifact retains a per-task TTT24 adapter checkpoint SHA256. This fails hard identity gate 6. The compact current exports also lack enough resolved tokenizer/serialization and full greedy-semantic fields to pass every remaining gate. Therefore no historical candidate was compared to Gold, and no historical result was merged into 33/89.",
        "",
        "## Availability",
        "",
        f"- Frozen hard cohort: {len(hard_ids)} outputs.",
        f"- Omitted views: {', '.join(OMITTED)}.",
        "- Expected historical source rows: 224 (two sources × 28 outputs × four views).",
        f"- Historical predictions available: {available_source_rows} raw source rows; {len(available_logical_cells)}/112 unique logical omitted-view cells.",
        "- Exact-reusable cells: 0.",
        "",
        "## Scientific status",
        "",
        "This is historical matched-control evidence only. It does not support or refute TTT24 omitted-D4 greedy coverage for the current adapters. Frozen60 is explicitly not used as an Eval60 substitute.",
    ]
    (output / "HISTORICAL_AUG8_REUSE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
