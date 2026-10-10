"""Frozen, target-blind contract helpers for E04 V3 fixed-64 localization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3"
CONFIG = OUT / "E04_V3_FIXED64_BATCH_RUNG_LOCALIZATION_V2_CONFIG.json"
PROTOCOL_ID = "E04_V3_FIXED64_BATCH_RUNG_LOCALIZATION_V2_MATCHED_FRESH_B1"
ARMS = (16, 8, 4, 1)
CAP_SECONDS = 1200


class LocalizationFailure(RuntimeError):
    pass


def sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    os.replace(temp, path)


def load_and_validate_config(path: Path = CONFIG) -> dict[str, Any]:
    config = read_json(path)
    required = {
        "schema_version", "protocol_id", "source_protocol_id", "cohort_path", "cohort_sha256",
        "prompt_path", "prompt_sha256", "checkpoint_manifest_path", "checkpoint_manifest_sha256",
        "native_config_provenance_path", "native_config_provenance_sha256",
        "runtime_identity_path", "runtime_identity_sha256", "native_config_dir", "seed",
        "max_new_tokens", "context_window", "batch_arms", "reference_arm", "hard_runtime_cap_seconds",
        "target_sidecar_path", "target_sidecar_sha256", "forbidden",
    }
    if set(config) != required or config["protocol_id"] != PROTOCOL_ID:
        raise LocalizationFailure("E04_FIXED64_CONFIG_SCHEMA")
    if config["batch_arms"] != list(ARMS) or config["reference_arm"] != 1 or config["hard_runtime_cap_seconds"] != CAP_SECONDS:
        raise LocalizationFailure("E04_FIXED64_ARM_OR_CAP")
    if config["forbidden"] != ["NO_TRAINING", "NO_OPTIMIZER", "NO_BACKWARD", "NO_GRADIENT", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT"]:
        raise LocalizationFailure("E04_FIXED64_FORBIDDEN_CONTRACT")
    for path_key, hash_key in (("cohort_path", "cohort_sha256"), ("prompt_path", "prompt_sha256"),
                               ("checkpoint_manifest_path", "checkpoint_manifest_sha256"),
                               ("native_config_provenance_path", "native_config_provenance_sha256"),
                               ("runtime_identity_path", "runtime_identity_sha256")):
        candidate = ROOT / config[path_key]
        if not candidate.is_file() or sha_path(candidate) != config[hash_key]:
            raise LocalizationFailure("E04_FIXED64_IDENTITY:" + path_key)
    cohort = read_json(ROOT / config["cohort_path"])
    prompts = read_jsonl(ROOT / config["prompt_path"])
    indexes = cohort.get("row_indexes")
    if not isinstance(indexes, list) or len(indexes) != 64 or len(set(indexes)) != 64 or any(not isinstance(v, int) or v < 0 or v >= len(prompts) for v in indexes):
        raise LocalizationFailure("E04_FIXED64_COHORT")
    # Target file identity is fixed but target bytes remain unopened here.
    if not (ROOT / config["target_sidecar_path"]).is_file():
        raise LocalizationFailure("E04_FIXED64_TARGET_SIDECAR_MISSING")
    return config


RAW_FIELDS = {"row_index", "input_sha256", "generated_token_ids", "text", "parser_valid", "parsed_grid", "prompt_tokens", "completion_tokens", "elapsed_seconds", "effective_batch_size", "physical_returned_token_ids", "pad_token_count"}


def validate_arm_raw(path: Path, indexes: list[int], prompt_hashes: dict[str, str], arm: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if [row.get("row_index") for row in rows] != indexes:
        raise LocalizationFailure("E04_FIXED64_ARM_COVERAGE")
    for row in rows:
        if set(row) != RAW_FIELDS or row["effective_batch_size"] != arm:
            raise LocalizationFailure("E04_FIXED64_ARM_SCHEMA")
        if not isinstance(row["generated_token_ids"], list) or not all(isinstance(v, int) for v in row["generated_token_ids"]):
            raise LocalizationFailure("E04_FIXED64_RAW_TOKEN_CAPTURE")
        if row["completion_tokens"] != len(row["generated_token_ids"]) or row["physical_returned_token_ids"][:len(row["generated_token_ids"])] != row["generated_token_ids"]:
            raise LocalizationFailure("E04_FIXED64_RAW_TOKEN_BOUNDARY")
        if row["parser_valid"] != (row["parsed_grid"] is not None) or prompt_hashes.get(str(row["row_index"])) != row["input_sha256"]:
            raise LocalizationFailure("E04_FIXED64_ARM_PARSE_OR_INPUT")
    return rows


def compare_non_target(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[int]:
    return [a["row_index"] for a, b in zip(reference, candidate, strict=True)
            if a["generated_token_ids"] != b["generated_token_ids"] or a["parser_valid"] != b["parser_valid"] or a["parsed_grid"] != b["parsed_grid"]]
