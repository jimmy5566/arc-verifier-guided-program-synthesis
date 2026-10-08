"""Remote-native, CPU-only sealed reference mapping for Directive 022.

This program reads only the Director-authorized, non-Gold source shards. It
never loads a model or tokenizer and emits no prompt/reference payloads: the
mapping artifact contains only identifiers and hashes and belongs in the sealed
scorer workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from generation_capability_audit_v1.audit import (
    SOURCE_COMMIT as SERIALIZER_SOURCE_COMMIT,
    build_generation_example,
    canonical_sha256,
    context_audit_record,
)

EXPECTED_REPLAY_SHA256 = "32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc"
SERIALIZER_VERSION = "generation_capability_audit_v1.audit:build_generation_example@" + SERIALIZER_SOURCE_COMMIT
SURFACES = ("NOVEL", "REPLAY", "PROTECTED")

class MappingError(RuntimeError): pass

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)

def source_token_label_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps({"input_ids": row["input_ids"], "labels": row["labels"]}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def read_manifest_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items: raise MappingError(f"INVALID_INPUT_MANIFEST={path}")
    if payload.get("contains_targets") is not False: raise MappingError(f"INPUT_MANIFEST_NOT_TARGET_BLIND={path}")
    return items

def selected_rows(paths: list[Path], sample_ids: set[str]) -> dict[str, tuple[dict[str, Any], Path]]:
    remaining, found = set(sample_ids), {}
    for path in paths:
        if not remaining: break
        table = pq.read_table(path, columns=["sample_id", "input_ids", "labels"], filters=[("sample_id", "in", sorted(remaining))])
        for row in table.to_pylist():
            sample_id = str(row["sample_id"])
            if sample_id in found: raise MappingError(f"AMBIGUOUS_SAMPLE_ID={sample_id}")
            found[sample_id] = (row, path)
            remaining.discard(sample_id)
    if remaining: raise MappingError("SELECTED_ROWS_MISSING=" + ",".join(sorted(remaining)))
    return found

def verified_novel_paths(root: Path, manifest: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = json.loads(manifest.read_text(encoding="utf-8")); shards = payload.get("shards")
    if payload.get("status") != "PASS" or not isinstance(shards, list) or len(shards) != 9: raise MappingError("NOVEL_SHARD_MANIFEST_INVALID")
    verified, identities = [], []
    for shard in shards:
        logical = str(shard["logical_name"]); path = root / Path(logical).name; actual = sha256_file(path); parquet = pq.ParquetFile(path)
        if actual != shard["sha256"]: raise MappingError(f"NOVEL_SHARD_SHA256_MISMATCH={path}")
        if path.stat().st_size != int(shard["bytes"]): raise MappingError(f"NOVEL_SHARD_BYTES_MISMATCH={path}")
        if parquet.metadata.num_rows != int(shard["rows"]): raise MappingError(f"NOVEL_SHARD_ROWS_MISMATCH={path}")
        verified.append(path); identities.append({"logical_name": logical, "path": str(path), "sha256": actual, "bytes": path.stat().st_size, "rows": parquet.metadata.num_rows})
    return verified, identities

def map_surface(name: str, items: list[dict[str, Any]], paths: list[Path], source_hashes: dict[str, str], source_repository: str, source_commit: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {str(item["sample_id"]) for item in items}
    if len(expected) != len(items): raise MappingError(f"DUPLICATE_MANIFEST_ITEM={name}")
    rows, mapping = selected_rows(paths, expected), []
    for ordinal, item in enumerate(items):
        sample_id = str(item["sample_id"]); row, source_path = rows[sample_id]; observed_label_hash = source_token_label_sha256(row)
        if observed_label_hash != item["token_label_sha256"]: raise MappingError(f"TOKEN_LABEL_SHA256_MISMATCH={name}:{sample_id}")
        example, audit = build_generation_example(row, item), None
        audit = context_audit_record(example, row)
        if not (audit["prompt_is_exact_source_prefix"] and audit["target_gold_is_exact_disjoint_source_suffix"] and audit["no_later_tokens"]): raise MappingError(f"PROMPT_TARGET_COMPATIBILITY_FAILED={name}:{sample_id}")
        mapping.append({
            "manifest_item_id": sample_id, "ordinal": ordinal, "source_repository": source_repository, "source_commit": source_commit,
            "source_shard_path": str(source_path), "source_shard_sha256": source_hashes[str(source_path)], "row_key": sample_id,
            "canonical_input_prompt_sha256": audit["prompt_sha256"], "canonical_reference_output_grid_sha256": canonical_sha256(example.gold_grid),
            "reference_output_dimensions": [len(example.gold_grid), len(example.gold_grid[0])], "token_label_sha256": observed_label_hash,
            "serializer_version": SERIALIZER_VERSION,
            "mapping_proof": {"prompt_is_exact_source_prefix": True, "target_gold_is_exact_disjoint_source_suffix": True, "target_eos_at_boundary": audit["target_eos_at_boundary"], "no_later_tokens": True, "target_labels_never_passed_to_generation": True},
        })
    actual = {entry["manifest_item_id"] for entry in mapping}
    if actual != expected or len(mapping) != len(expected): raise MappingError(f"BIJECTION_FAILURE={name}")
    return mapping, {"item_count": len(mapping), "unique_manifest_item_count": len(actual), "mapping_sha256": canonical_sha256(mapping), "token_label_hashes_verified": len(mapping), "exact_bijection": True}

def main() -> int:
    parser = argparse.ArgumentParser()
    for flag in ("novel-root", "replay-shard", "novel-shard-manifest", "input-manifest-dir", "sealed-output", "receipt"): parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--source-repository", required=True); parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        novel_paths, novel_identities = verified_novel_paths(args.novel_root, args.novel_shard_manifest)
        replay_hash = sha256_file(args.replay_shard); replay_rows = pq.ParquetFile(args.replay_shard).metadata.num_rows
        if replay_hash != EXPECTED_REPLAY_SHA256: raise MappingError("REPLAY_SHA256_MISMATCH")
        if replay_rows != 62650: raise MappingError("REPLAY_ROWS_MISMATCH")
        manifest_paths = {"NOVEL": args.input_manifest_dir / "NOVEL_GENERATION_INPUT_MANIFEST.json", "REPLAY": args.input_manifest_dir / "REPLAY_GENERATION_INPUT_MANIFEST.json", "PROTECTED": args.input_manifest_dir / "PROTECTED_CAPABILITY_INPUT_MANIFEST.json"}
        input_manifest_hashes = {name: sha256_file(path) for name, path in manifest_paths.items()}; items = {name: read_manifest_items(path) for name, path in manifest_paths.items()}
        source_hashes = {entry["path"]: entry["sha256"] for entry in novel_identities}; source_hashes[str(args.replay_shard)] = replay_hash
        surfaces, summaries = {}, {}
        surfaces["NOVEL"], summaries["NOVEL"] = map_surface("NOVEL", items["NOVEL"], novel_paths, source_hashes, args.source_repository, args.source_commit)
        surfaces["REPLAY"], summaries["REPLAY"] = map_surface("REPLAY", items["REPLAY"], [args.replay_shard], source_hashes, args.source_repository, args.source_commit)
        surfaces["PROTECTED"], summaries["PROTECTED"] = map_surface("PROTECTED", items["PROTECTED"], [args.replay_shard], source_hashes, args.source_repository, args.source_commit)
        required = {"NOVEL": 128, "REPLAY": 64, "PROTECTED": 64}
        if {name: summaries[name]["item_count"] for name in SURFACES} != required: raise MappingError("DIRECTIVE_022_CARDINALITY_MISMATCH")
        artifact = {"schema_version": 1, "directive_id": "DIRECTOR_DIRECTIVE_022", "protocol_id": "ROUND_009_POST_HOC_CAPABILITY_CHARACTERIZATION_V1", "round_id": "RECONSTRUCTED_FOUNDATION_V2_V2_009", "sealed_scorer_workspace_only": True, "generation_side_contains_references": False, "serializer_version": SERIALIZER_VERSION, "source_repository": args.source_repository, "source_commit": args.source_commit, "input_manifest_hashes": input_manifest_hashes, "novel_shards": novel_identities, "replay_shard": {"path": str(args.replay_shard), "sha256": replay_hash, "rows": replay_rows}, "surfaces": surfaces, "summaries": summaries, "cpu_only": True, "model_loaded": False, "generation_started": False, "scoring_started": False, "scientific_training_started": False, "final_audit_opened": False}
        atomic_json(args.sealed_output, artifact); output_hash = sha256_file(args.sealed_output)
        receipt = {"schema_version": 1, "directive_id": "DIRECTOR_DIRECTIVE_022", "round_id": "RECONSTRUCTED_FOUNDATION_V2_V2_009", "mapping_artifact_path": str(args.sealed_output), "mapping_artifact_sha256": output_hash, "row_item_counts": {name: summaries[name]["item_count"] for name in SURFACES}, "input_manifest_hashes": input_manifest_hashes, "source_hashes": {"replay": replay_hash, "novel_shards": {item["logical_name"]: item["sha256"] for item in novel_identities}}, "exit_status": "SUCCESS", "cpu_only": True, "model_loaded": False, "generation_started": False, "scoring_started": False, "scientific_training_started": False, "final_audit_opened": False}
        atomic_json(args.receipt, receipt); print(json.dumps(receipt, sort_keys=True)); return 0
    except Exception as exc:
        failure = {"schema_version": 1, "directive_id": "DIRECTOR_DIRECTIVE_022", "exit_status": "FAILURE", "error_type": type(exc).__name__, "error": str(exc), "cpu_only": True, "model_loaded": False, "generation_started": False, "scoring_started": False, "scientific_training_started": False, "final_audit_opened": False}
        atomic_json(args.receipt, failure); print(json.dumps(failure, sort_keys=True)); return 2

if __name__ == "__main__": raise SystemExit(main())