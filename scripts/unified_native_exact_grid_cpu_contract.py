"""CPU-only exact-grid scoring for the frozen native six-condition evidence.

The sealed V1 sidecar stores ARC grids under ``targets``.  It deliberately does
not store tokenizer continuations, so this module only scores already-generated
free-running predictions and never performs a model forward, ranking, or
generation.  Result records retain booleans and hashes, never target grids.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.arc2_token_grid_parser import TokenGridContract, parse_generated_token_ids

CONDITIONS = (
    "QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED",
    "RECONSTRUCTED_FOUNDATION_V2_V7",
    "TARGETED_R1",
    "TARGETED_R2",
    "ATOMIC_R1",
    "FAMILY_BALANCED",
)
FAMILIES = (
    "STRUCTURAL_TRANSLATION",
    "STRUCTURAL_REFLECTION",
    "COMPOSITION_RECOLOR_TRANSLATE",
    "COMPOSITION_REFLECT_RECOLOR",
    "PROTECTED_SAME_COLOR",
)
# The native runtime verified this contract before producing the frozen rows.
# It is revalidated below by reconstructing every saved canonical prediction
# hash; this CPU phase never guesses a tokenizer mapping.
FROZEN_NATIVE_GRID_CONTRACT = TokenGridContract(tuple(range(10)), 10, 15, 13)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_grid_hash(grid: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps({"grid": grid}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _grid_hash_from_tokens(token_ids: list[int], *, label: str) -> str:
    parsed = parse_generated_token_ids(token_ids, FROZEN_NATIVE_GRID_CONTRACT)
    if parsed.grid is None:
        raise RuntimeError(f"{label}_TOKEN_GRID_INVALID:{parsed.parse_reason}")
    return canonical_grid_hash(parsed.grid)


def read_frozen_raw(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError("RAW_EVIDENCE_IDENTITY_MISMATCH")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 360:
        raise RuntimeError("RAW_ROW_COUNT_MISMATCH")
    forbidden = {"target", "targets", "output", "gold", "dGold", "final_audit"}
    keys = {(str(row.get("checkpoint_condition")), str(row.get("episode_id"))) for row in rows}
    counts = Counter(str(row.get("checkpoint_condition")) for row in rows)
    if len(keys) != 360 or set(counts) != set(CONDITIONS) or any(counts[condition] != 60 for condition in CONDITIONS):
        raise RuntimeError("RAW_CONDITION_COMPLETENESS_INVALID")
    for row in rows:
        if forbidden & set(row):
            raise RuntimeError("RAW_TARGET_LEAK")
        if not isinstance(row.get("episode_id"), str) or not isinstance(row.get("family"), str):
            raise RuntimeError("RAW_EVIDENCE_SCHEMA_INVALID")
        if not isinstance(row.get("prompt_sha256"), str) or not isinstance(row.get("generated_token_ids"), list):
            raise RuntimeError("RAW_EVIDENCE_SCHEMA_INVALID")
        if not row.get("parse_valid") or row.get("parse_reason") is not None:
            raise RuntimeError("RAW_GENERATION_PARSE_NOT_FROZEN_VALID")
        reconstructed = _grid_hash_from_tokens(row["generated_token_ids"], label="RAW_GREEDY")
        if reconstructed != row.get("canonical_prediction_sha256"):
            raise RuntimeError("RAW_CANONICAL_PREDICTION_RECONSTRUCTION_MISMATCH")
    family_by_episode = {row["episode_id"]: row["family"] for row in rows}
    if len(family_by_episode) != 60 or set(family_by_episode.values()) != set(FAMILIES):
        raise RuntimeError("RAW_FAMILY_MAPPING_INVALID")
    if any(sum(1 for family in family_by_episode.values() if family == value) != 12 for value in FAMILIES):
        raise RuntimeError("RAW_FAMILY_DENOMINATOR_INVALID")
    return rows


def read_grid_sidecar_hashes(path: Path, expected_sha256: str, episode_ids: set[str]) -> dict[str, str]:
    """Validate the frozen grid schema and return target hashes only."""
    if not path.is_file():
        raise RuntimeError("SEALED_TARGET_SIDECAR_MISSING")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("SEALED_TARGET_SIDECAR_SHA_MISMATCH")
    document = json.loads(path.read_text(encoding="utf-8"))
    targets = document.get("targets")
    if not isinstance(targets, dict) or set(targets) != episode_ids:
        raise RuntimeError("SEALED_TARGET_SIDECAR_EPISODE_MAPPING_INVALID")
    result: dict[str, str] = {}
    for episode_id, grid in targets.items():
        if not isinstance(episode_id, str) or not isinstance(grid, list) or not grid:
            raise RuntimeError("SEALED_TARGET_SIDECAR_GRID_SCHEMA_INVALID")
        width: int | None = None
        for row in grid:
            if not isinstance(row, list) or not row:
                raise RuntimeError("SEALED_TARGET_SIDECAR_GRID_SCHEMA_INVALID")
            if width is None:
                width = len(row)
            if len(row) != width or any(type(value) is not int or value < 0 or value > 9 for value in row):
                raise RuntimeError("SEALED_TARGET_SIDECAR_GRID_SCHEMA_INVALID")
        result[episode_id] = canonical_grid_hash(grid)
    return result


def score_exact_grid_rows(rows: list[dict[str, Any]], target_hashes: dict[str, str]) -> list[dict[str, Any]]:
    episode_ids = {row["episode_id"] for row in rows}
    if set(target_hashes) != episode_ids:
        raise RuntimeError("TARGET_ALIGNMENT_EPISODE_MAPPING_INVALID")
    scored: list[dict[str, Any]] = []
    for row in rows:
        greedy_hash = row["canonical_prediction_sha256"]
        alternate_hash: str | None = None
        alternate = row.get("alternate_token_ids")
        if not isinstance(alternate, list) or not alternate:
            raise RuntimeError("RAW_ALTERNATE_EVIDENCE_MISSING")
        parsed_alternate = parse_generated_token_ids(alternate, FROZEN_NATIVE_GRID_CONTRACT)
        if parsed_alternate.grid is not None:
            alternate_hash = canonical_grid_hash(parsed_alternate.grid)
        target_hash = target_hashes[row["episode_id"]]
        greedy = greedy_hash == target_hash
        alternate_only = (not greedy) and alternate_hash == target_hash
        scored.append({
            "checkpoint_condition": row["checkpoint_condition"],
            "episode_id": row["episode_id"],
            "family": row["family"],
            "prompt_sha256": row["prompt_sha256"],
            "greedy_exact_grid_match": greedy,
            "alternate_only_exact_grid_match": alternate_only,
            "complete_output_top2_coverage": greedy or alternate_only,
            "greedy_prediction_sha256": greedy_hash,
            "alternate_prediction_sha256": alternate_hash,
            "target_grid_sha256": target_hash,
            "target_alignment_used_for_selection": False,
        })
    return scored


def _point(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise RuntimeError("EXACT_DENOMINATOR_ZERO")
    return {
        "denominator": len(rows),
        "greedy_top1_exact": sum(bool(row["greedy_exact_grid_match"]) for row in rows),
        "alternate_only_exact": sum(bool(row["alternate_only_exact_grid_match"]) for row in rows),
        "complete_output_top2_coverage": sum(bool(row["complete_output_top2_coverage"]) for row in rows),
    }


def summarize_exact_grid(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scored) != 360:
        raise RuntimeError("EXACT_RECORD_COUNT_INVALID")
    by_condition = {condition: _point([row for row in scored if row["checkpoint_condition"] == condition]) for condition in CONDITIONS}
    by_family = {family: _point([row for row in scored if row["family"] == family]) for family in FAMILIES}
    if any(point["denominator"] != 60 for point in by_condition.values()) or any(point["denominator"] != 72 for point in by_family.values()):
        raise RuntimeError("EXACT_SUMMARY_DENOMINATOR_INVALID")
    return {"pooled": _point(scored), "by_condition": by_condition, "by_family": by_family}
