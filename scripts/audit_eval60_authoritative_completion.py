#!/usr/bin/env python3
"""Read-only integrity audit for the Eval60 authoritative overnight run.

This does not score candidates or open a solutions file.  It is deliberately
usable after a partial stop so that the morning handoff distinguishes retained
evidence from a fully complete authoritative dataset.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


EXPECTED_TABLES = (
    "01_cells.parquet", "02_greedy_tokens.parquet", "03_greedy_candidates.parquet",
    "04_turbodfs_candidates.parquet", "05_turbodfs_nodes.parquet", "06_runtime.parquet",
    "07_cross_aug_features.parquet", "08_cross_depth_features.parquet", "09_gold_labels.parquet",
    "10_checkpoint_map.csv", "11_decoder_config.json",
)
TOKEN_FIELDS = {
    "token_index", "chosen_token_id", "chosen_token_logprob", "top1_token_id",
    "top1_prob", "top2_token_id", "top2_prob", "top1_minus_top2_margin",
    "entropy", "cumulative_sequence_logprob", "full_native_logprobs",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    identities = manifest.get("identity", {})

    task_status: list[str] = []
    for path in sorted((root / "task_status").glob("*.json")):
        try:
            task_status.append(str(read_json(path).get("status", "UNKNOWN")))
        except Exception:
            task_status.append("UNREADABLE")
    status_counts = dict(collections.Counter(task_status))

    checkpoint_failures: list[str] = []
    adapter_count = 0
    for adapter in sorted((root / "checkpoints").glob("*/depth_*/adapter_model.safetensors")):
        adapter_count += 1
        metadata_path = adapter.with_name("metadata.json")
        if adapter.stat().st_size <= 0 or not metadata_path.is_file():
            checkpoint_failures.append(str(adapter))
            continue
        metadata = read_json(metadata_path)
        if metadata.get("checkpoint_sha256") != sha256(adapter):
            checkpoint_failures.append(str(adapter))

    cells = sorted((root / "raw" / "greedy_cells").rglob("*.json"))
    bad_cells: list[str] = []
    missing_token_fields = 0
    noncompact_vectors = 0
    for path in cells:
        try:
            row = read_json(path)
            tokens = row.get("token_telemetry")
            if not isinstance(tokens, list) or not tokens:
                bad_cells.append(str(path)); continue
            for token in tokens:
                if not TOKEN_FIELDS.issubset(token):
                    missing_token_fields += 1
                vector = token.get("full_native_logprobs")
                if not isinstance(vector, list) or len(vector) < 8:
                    noncompact_vectors += 1
        except Exception:
            bad_cells.append(str(path))

    freeze_path = root / "GREEDY_GENERATION_FROZEN.flag"
    freeze = read_json(freeze_path) if freeze_path.is_file() else {}
    greedy_gold = read_json(root / "greedy_gold_summary.json") if (root / "greedy_gold_summary.json").is_file() else {}
    turbo_freeze_path = root / "TURBODFS_GENERATION_FROZEN.flag"
    turbo_freeze = read_json(turbo_freeze_path) if turbo_freeze_path.is_file() else {}
    analysis = root / "analysis_ready"
    table_status = {name: (analysis / name).is_file() for name in EXPECTED_TABLES}
    disk = shutil.disk_usage(root)

    report: dict[str, Any] = {
        "run_directory": str(root),
        "task_status": status_counts,
        "tasks_complete": status_counts.get("GREEDY_COMPLETE", 0),
        "manifest_tasks": identities.get("tasks"),
        "manifest_outputs": identities.get("outputs"),
        "adapters_retained": adapter_count,
        "adapters_hash_valid": adapter_count - len(checkpoint_failures),
        "checkpoint_failures": checkpoint_failures,
        "greedy_cells": len(cells),
        "bad_greedy_cells": bad_cells,
        "token_fields_missing": missing_token_fields,
        "noncompact_probability_vectors": noncompact_vectors,
        "greedy_frozen": freeze_path.is_file(),
        "gold_accessed_pre_greedy_freeze": freeze.get("gold_accessed_pre_freeze"),
        "greedy_gold_summary": greedy_gold,
        "turbodfs_frozen": turbo_freeze_path.is_file(),
        "gold_accessed_pre_turbo_freeze": turbo_freeze.get("turbo_gold_accessed_before_freeze"),
        "analysis_ready_tables": table_status,
        "data_dictionary": (root / "DATA_DICTIONARY.md").is_file(),
        "retention_manifest": (root / "DATA_RETENTION_MANIFEST.csv").is_file(),
        "checkpoint_retention_readme": (root / "CHECKPOINT_RETENTION_README.md").is_file(),
        "morning_handoff": (root / "MORNING_HANDOFF.md").is_file(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }
    complete = (
        report["tasks_complete"] == 60 and identities.get("tasks") == 60 and
        identities.get("outputs") == 89 and adapter_count == 180 and
        not checkpoint_failures and len(cells) == 1068 and not bad_cells and
        missing_token_fields == 0 and noncompact_vectors == 0 and
        freeze.get("gold_accessed_pre_freeze") is False
    )
    report["authoritative_greedy_complete"] = complete
    target = args.report or (root / "authoritative_completion_audit.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target), "authoritative_greedy_complete": complete}, sort_keys=True))
    if args.strict and not complete:
        sys.exit(2)


if __name__ == "__main__":
    main()
