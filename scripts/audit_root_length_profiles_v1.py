#!/usr/bin/env python3
"""Tokenizer-only target-blind root-length audit for frozen AUG16 profiles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.nvarc_native import checkpoint_native_tokenizer  # noqa: E402
from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix  # noqa: E402
from inference.root_length_memory_profile import (  # noqa: E402
    KV_BLOCK_TOKENS,
    MAX_NEW_TOKENS,
    deterministic_resident_groups,
    required_capacity_for_root,
    select_memory_profile,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only  # noqa: E402
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import _load_aug16, _transform_task  # noqa: E402


EXPERIMENT = "ROOT_LENGTH_PROFILE_AUDIT_V1"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outputs_from_manifest(path: Path) -> list[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = list(payload.get("eligible_output_ids", []))
    if payload.get("status") != "PREPARED_TARGET_BLIND" or not bool(ids):
        raise RuntimeError("frozen target-blind eligible-output manifest is required")
    parsed: list[tuple[str, int]] = []
    for output_id in ids:
        task_id, marker, index = str(output_id).partition(":o")
        if marker != ":o" or not task_id or not index.isdecimal():
            raise RuntimeError(f"malformed output ID in frozen manifest: {output_id}")
        parsed.append((task_id, int(index)))
    if len(parsed) != len(set(parsed)):
        raise RuntimeError("frozen output manifest contains a duplicate output")
    return parsed


def _quantile(values: list[int], point: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(point * 100) - 1]) if len(values) > 1 else float(values[0])


def _native_prompt_length(*, tokenizer: Any, task: Any, output_index: int, candidate: dict[str, Any]) -> int:
    """Tokenize the exact frozen native prompt without loading a model.

    The Step-0 public text helper intentionally addresses only ``test[0]``;
    this audit covers multi-output tasks and must instead bind the selected
    public test input through the native formatter itself.
    """
    transformed = _transform_task(task, output_index, candidate)
    prefix = native_training_message_prefix(transformed)
    messages = native_messages_from_training_prefix(prefix, transformed.test[0].input)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return int(encoded["input_ids"].shape[-1])


def run(args: argparse.Namespace) -> None:
    _assert_challenge_only(args.challenge)
    raw = json.loads(args.challenge.read_text(encoding="utf-8")); tasks = load_dataset(args.challenge)
    outputs = _outputs_from_manifest(args.output_manifest)
    candidates = _load_aug16(args.candidate_pool, args.aug16_ids)
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    rows: list[dict[str, Any]] = []; assignments: dict[str, Any] = {}
    for task_id, output_index in outputs:
        if task_id not in tasks or task_id not in raw or output_index >= len(tasks[task_id].test):
            raise RuntimeError(f"frozen output unavailable in target-blind challenge: {task_id}:o{output_index}")
        lengths: list[int] = []
        for candidate in candidates:
            lengths.append(_native_prompt_length(tokenizer=tokenizer, task=tasks[task_id], output_index=output_index,
                                                 candidate=candidate))
        root_max = max(lengths); profile = select_memory_profile(root_max); groups = deterministic_resident_groups(candidate_ids, profile)
        row = {"output_id": f"{task_id}:o{output_index}", "task_id": task_id, "output_index": output_index,
               "root_length_min": min(lengths), "root_length_max": root_max, "root_length_mean": statistics.fmean(lengths),
               "unique_prompt_lengths": len(set(lengths)), "profile": profile.name, "resident_width": profile.resident_width,
               "physical_batch_ceiling": profile.physical_batch_ceiling, "resident_group_count": len(groups),
               "initial_kv_capacity": ((root_max + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS,
               "final_required_kv_capacity": required_capacity_for_root(root_max),
               **{f"aug{index:02d}_token_length": length for index, length in enumerate(lengths)}}
        rows.append(row)
        assignments[row["output_id"]] = {**row, "augmentation_ids": candidate_ids,
                                          "resident_groups": [list(group) for group in groups]}
    rows.sort(key=lambda row: str(row["output_id"]))
    maxima = [int(row["root_length_max"]) for row in rows]
    counts = {profile: sum(row["profile"] == profile for row in rows) for profile in ("PROFILE_S", "PROFILE_M", "PROFILE_L", "PROFILE_XL", "PROFILE_XXL", "PROFILE_OVERSIZE")}
    fields = list(rows[0])
    _atomic_csv(args.output / "OUTPUT_ROOT_LENGTHS.csv", rows, fields)
    _atomic_csv(args.output / "PROFILE_COUNTS.csv", [{"profile": profile, "output_count": count} for profile, count in counts.items()], ["profile", "output_count"])
    audit = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "model_loaded": False,
             "frozen_output_manifest": str(args.output_manifest), "frozen_output_manifest_sha256": _sha256_file(args.output_manifest),
             "challenge_sha256": _sha256_file(args.challenge), "candidate_pool_sha256": _sha256_file(args.candidate_pool),
             "aug16_ids_sha256": _sha256_file(args.aug16_ids), "tokenizer_identity": tokenizer_identity,
             "output_count": len(rows), "cohort_root_length_summary": {"min": min(maxima), "median": statistics.median(maxima),
             "p90": _quantile(maxima, 0.90), "max": max(maxima)}, "profile_counts": counts,
             "kv_block_tokens": KV_BLOCK_TOKENS, "max_new_tokens": MAX_NEW_TOKENS, "assignments": assignments}
    _atomic_json(args.output / "PROFILE_ASSIGNMENTS.json", audit)
    report = ["# Root-length profile audit V1", "", "Tokenizer-only, target-blind audit over the frozen 89-output Census manifest. No model, adapter, GPU inference, or Gold was loaded.", "",
              f"- Outputs: `{len(rows)}`", f"- Root max length min / median / p90 / max: `{min(maxima)}` / `{statistics.median(maxima)}` / `{_quantile(maxima, 0.90)}` / `{max(maxima)}`", "", "## Frozen profiles", ""]
    report.extend(f"- `{name}`: `{count}`" for name, count in counts.items())
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {path.name: _sha256_file(path) for path in args.output.iterdir() if path.is_file() and path.name != "HASHES.json"}
    _atomic_json(args.output / "HASHES.json", {"algorithm": "sha256", "files": hashes})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True); parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True); parser.add_argument("--aug16-ids", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
