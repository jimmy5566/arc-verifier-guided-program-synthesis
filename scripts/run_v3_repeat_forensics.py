"""Train-only, task-generic forensics for V3 REPEAT semantic failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.evidence.extractor import _background, extract_task_evidence


def _component_signature(item: Any) -> tuple[int, tuple[tuple[int, int], ...]]:
    top, left, _bottom, _right = item.bbox
    return item.color, tuple((row - top, col - left) for row, col in item.cells)


def _pair_forensics(pair: Any) -> dict[str, object]:
    """Evidence labels, not oracle claims, for one input/output train pair."""
    source, target = pair.input_grid, pair.output_grid
    input_bg, output_bg = _background(source), _background(target)
    source_signatures = Counter(_component_signature(item) for item in pair.input_objects)
    target_signatures = Counter(_component_signature(item) for item in pair.output_objects)
    shared = sum((source_signatures & target_signatures).values())
    source_colors, target_colors = {item.color for item in pair.input_objects}, {item.color for item in pair.output_objects}
    labels: set[str] = set()
    if source.shape != target.shape: labels.add("TERMINATION_ALIGNMENT_REFERENCE")
    if input_bg != output_bg: labels.add("COMPOSITION_STATE_UPDATE_SEMANTICS")
    if not shared: labels.add("MOTIF_EXTRACTION_OR_TRANSFORM_UNRESOLVED")
    if source_colors - target_colors or target_colors - source_colors: labels.add("COLOR_OR_SHAPE_SEQUENCE")
    if len(target_signatures) > len(source_signatures): labels.add("PROGRESSIVE_SPACING_OR_STATE_UPDATE")
    if pair.parameter_candidates.get("DIRECTION") and len(pair.parameter_candidates["DIRECTION"]) > 1:
        labels.add("DIRECTION_AMBIGUITY")
    if pair.parameter_candidates.get("STEP") and len(pair.parameter_candidates["STEP"]) > 1:
        labels.add("STEP_OR_GAP_AMBIGUITY")
    if "COMPOSITION_STATE_UPDATE_SEMANTICS" not in labels and "TERMINATION_ALIGNMENT_REFERENCE" not in labels:
        labels.add("BOUNDARY_OR_COLLISION_TERMINATION_UNRESOLVED")
    return {
        "input_shape": list(source.shape), "output_shape": list(target.shape),
        "input_background": input_bg, "output_background": output_bg,
        "input_object_count": len(pair.input_objects), "output_object_count": len(pair.output_objects),
        "shared_component_signature_count": shared, "labels": sorted(labels),
    }


def _primary(pair_records: list[Mapping[str, object]]) -> str:
    labels = Counter(label for record in pair_records for label in record["labels"])  # type: ignore[index]
    priority = (
        "COMPOSITION_STATE_UPDATE_SEMANTICS", "TERMINATION_ALIGNMENT_REFERENCE",
        "MOTIF_EXTRACTION_OR_TRANSFORM_UNRESOLVED", "COLOR_OR_SHAPE_SEQUENCE",
        "PROGRESSIVE_SPACING_OR_STATE_UPDATE", "DIRECTION_AMBIGUITY", "STEP_OR_GAP_AMBIGUITY",
        "BOUNDARY_OR_COLLISION_TERMINATION_UNRESOLVED",
    )
    return next(label for label in priority if labels[label])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--oracle-semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen forensics")
    cohort, oracle = (json.loads(path.read_text(encoding="utf-8")) for path in (args.cohort, args.oracle_semantic))
    ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    if len(ids) != 30 or task_hash != oracle["task_ids_hash"]: raise ValueError("requires exact frozen30")
    tasks, records = load_dataset(args.challenge_path), {}
    for task_id in ids:
        if "REPEAT" not in oracle["gold"][task_id]["operations"]: continue
        evidence = extract_task_evidence(tasks[task_id])
        pairs = [_pair_forensics(item) for item in evidence.pairs]
        records[task_id] = {"primary": _primary(pairs), "pair_forensics": pairs}
    aggregate = Counter(item["primary"] for item in records.values())
    observed_labels = Counter(
        label
        for item in records.values()
        for label in {label for pair in item["pair_forensics"] for label in pair["labels"]}  # type: ignore[index]
    )
    taxonomy_vocabulary = (
        "MOTIF_EXTRACTION_OR_TRANSFORM_UNRESOLVED",
        "SOURCE_REFERENCE_ROLE_UNRESOLVED",
        "DIRECTION_AMBIGUITY",
        "STEP_OR_GAP_AMBIGUITY",
        "PROGRESSIVE_SPACING_OR_STATE_UPDATE",
        "COLOR_OR_SHAPE_SEQUENCE",
        "TERMINATION_BOUNDARY_UNRESOLVED",
        "TERMINATION_COLLISION_UNRESOLVED",
        "TERMINATION_ALIGNMENT_REFERENCE",
        "REPEAT_UNTIL_NO_CHANGE_UNRESOLVED",
        "COMPOSITION_STATE_UPDATE_SEMANTICS",
    )
    artifact = {
        "experiment_id": "ARC2_V3_REPEAT_SEMANTICS_EXPANSION",
        "status": "TRAIN_ONLY_REPEAT_FORENSICS_FROZEN",
        "task_ids_hash": task_hash,
        "repeat_failure_count_before": len(records),
        "protocol": "Private semantic oracle selects only REPEAT-family tasks. All evidence labels derive from train input/output pairs; no model, test grid/output, solution file or task-id branch is used.",
        "primary_taxonomy": dict(sorted(aggregate.items())),
        "observed_label_taxonomy": {label: observed_labels[label] for label in taxonomy_vocabulary},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"repeat_failures": len(records), "taxonomy": artifact["primary_taxonomy"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
