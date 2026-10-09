#!/usr/bin/env python3
"""Bounded, no-update teacher-forced rank/margin measurement worker.

It stays inert until the immutable launch binding passes every byte-identity
and runtime preflight.  Results contain aggregate rank/margin evidence only:
target grids and target token IDs never leave process memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.arc2_token_grid_parser import tokenizer_token_contract
from scripts.paired_rank_margin_launch_contract import CAP_SECONDS, atomic, failure_receipt, release_live_lock, require_launch
from scripts.paired_rank_margin_runtime_contract import extract_target_logit_indices, left_pad_teacher_forced, require_complete_paired_rows
from scripts.paired_v7_family_balanced_rank_margin_contract import require_layout, require_two_checkpoint_prompt_pairing, serialize_native_grid_target

PROTOCOL = "PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1"
CONDITIONS = ("RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED")
COMPONENTS = ("GRID_CONTENT", "ROW_SEPARATOR", "EOS_END")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_grid(grid: list[list[int]]) -> str:
    if not isinstance(grid, list) or not grid or any(not isinstance(row, list) or not row for row in grid):
        raise RuntimeError("INVALID_NATIVE_GRID")
    width = len(grid[0])
    if any(len(row) != width or any(type(value) is not int or not 0 <= value <= 9 for value in row) for row in grid):
        raise RuntimeError("INVALID_NATIVE_GRID")
    return "\n".join("".join(str(value) for value in row) for row in grid)


def native_prompt(task: dict) -> str:
    pieces: list[str] = []
    for pair in task["train"]:
        pieces.extend((f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>", f"<|im_start|>assistant\n{native_grid(pair['output'])}<|im_end|>"))
    for pair in task["test"]:
        if "output" in pair:
            raise RuntimeError("SEALED_TARGET_LEAK")
        pieces.append(f"<|im_start|>user\n{native_grid(pair['input'])}<|im_end|>")
    return "".join(pieces) + "<|im_start|>assistant\n"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"INPUT_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_manifest(manifest_path: Path, expected_manifest_sha256: str) -> dict:
    if sha(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("CHECKPOINT_MANIFEST_SHA_MISMATCH")
    manifest = load_json(manifest_path)
    for root_key, entries_key in (("base_path", "base_files"), ("adapter_path", "adapter_files")):
        root, entries = Path(manifest.get(root_key, "")), manifest.get(entries_key, [])
        if not root.is_dir() or not isinstance(entries, list) or not entries:
            raise RuntimeError("CHECKPOINT_MANIFEST_RUNTIME_INVALID")
        for entry in entries:
            candidate = root / entry.get("name", "")
            if not candidate.is_file() or candidate.stat().st_size != entry.get("bytes") or sha(candidate) != entry.get("sha256"):
                raise RuntimeError("CHECKPOINT_RUNTIME_FILE_MISMATCH")
    return manifest


def select_raw_pairs(raw_path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in records if row.get("checkpoint_condition") in CONDITIONS]
    prompts = require_two_checkpoint_prompt_pairing(rows)
    by_condition: dict[str, dict[str, dict]] = {condition: {} for condition in CONDITIONS}
    for row in rows:
        by_condition[row["checkpoint_condition"]][row["episode_id"]] = row
    return {episode_id: {"prompt_sha256": prompts[episode_id], "raw": {condition: by_condition[condition][episode_id] for condition in CONDITIONS}} for episode_id in prompts}


def validate_prompt_reconstruction(input_manifest: dict, raw_pairs: dict[str, dict]) -> None:
    """Byte-check all frozen prompts before tokenizer/model import."""
    episodes = input_manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 60:
        raise RuntimeError("PROMPT_RECONSTRUCTION_EPISODE_COUNT_INVALID")
    for episode in episodes:
        episode_id, task = episode.get("episode_id"), episode.get("observation", {}).get("task")
        if not isinstance(episode_id, str) or episode_id not in raw_pairs or not isinstance(task, dict):
            raise RuntimeError("PROMPT_RECONSTRUCTION_MAPPING_INVALID")
        if hashlib.sha256(native_prompt(task).encode("utf-8")).hexdigest() != raw_pairs[episode_id]["prompt_sha256"]:
            raise RuntimeError("PROMPT_RECONSTRUCTION_HASH_MISMATCH")


def runtime_rows(input_manifest: dict, sidecar: dict, raw_pairs: dict[str, dict], tokenizer) -> list[dict]:
    """Runtime-only target serialization after sidecar/map identity checks."""
    episodes, targets = input_manifest.get("episodes"), sidecar.get("targets")
    if not isinstance(episodes, list) or len(episodes) != 60 or not isinstance(targets, dict) or len(targets) != 60:
        raise RuntimeError("SEALED_RUNTIME_MAPPING_INVALID")
    validate_prompt_reconstruction(input_manifest, raw_pairs)
    contract = tokenizer_token_contract(tokenizer, eos_token_id=15, pad_token_id=13)
    output = []
    for episode in episodes:
        episode_id, task = episode.get("episode_id"), episode.get("observation", {}).get("task")
        if not isinstance(episode_id, str) or episode_id not in raw_pairs or episode_id not in targets or not isinstance(task, dict):
            raise RuntimeError("SEALED_RUNTIME_EPISODE_MAPPING_INVALID")
        prompt = native_prompt(task)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha256 != raw_pairs[episode_id]["prompt_sha256"]:
            raise RuntimeError("PROMPT_RECONSTRUCTION_HASH_MISMATCH")
        layout = serialize_native_grid_target(targets[episode_id], contract)
        require_layout(layout, contract)
        output.append({"episode_id": episode_id, "family": episode.get("family"), "prompt_ids": tokenizer(prompt, add_special_tokens=False)["input_ids"], "target_ids": list(layout.token_ids), "grid_positions": set(layout.grid_content_positions), "separator_positions": set(layout.row_separator_positions), "eos_positions": set(layout.eos_positions), "prompt_sha256": prompt_sha256, "free_running": raw_pairs[episode_id]["raw"]})
    families = {row["family"] for row in output}
    if len(output) != 60 or len({row["episode_id"] for row in output}) != 60 or len(families) != 5 or any(sum(row["family"] == family for row in output) != 12 for family in families):
        raise RuntimeError("RUNTIME_FAMILY_DENOMINATOR_INVALID")
    return output


def length_bucketed(rows: list[dict], size: int) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda row: (len(row["prompt_ids"]) + len(row["target_ids"]), row["episode_id"]))
    return [ordered[index:index + size] for index in range(0, len(ordered), size)]


def component_name(row: dict, target_index: int) -> str:
    matches = [name for name, positions in zip(COMPONENTS, (row["grid_positions"], row["separator_positions"], row["eos_positions"])) if target_index in positions]
    if len(matches) != 1:
        raise RuntimeError("TARGET_COMPONENT_PARTITION_INVALID")
    return matches[0]


def first_free_running_error(row: dict, condition: str) -> tuple[int | None, str]:
    """Find each checkpoint's own frozen free-running error location."""
    frozen = row["free_running"].get(condition)
    if not isinstance(frozen, dict):
        raise RuntimeError("FROZEN_GENERATION_CONDITION_INVALID")
    generated = frozen.get("generated_token_ids")
    if not isinstance(generated, list):
        raise RuntimeError("FROZEN_GENERATION_EVIDENCE_INVALID")
    parser_status = "PARSE_VALID" if frozen.get("parse_valid") is True else f"PARSE_INVALID:{frozen.get('parse_reason') or 'UNSPECIFIED'}"
    for index, target in enumerate(row["target_ids"]):
        if index >= len(generated) or generated[index] != target:
            return index, parser_status
    return (None if len(generated) == len(row["target_ids"]) else len(row["target_ids"])), parser_status


def score_batches(model, rows: list[dict], torch, condition: str, mode_id: str, deadline: float) -> list[dict]:
    batch_size, results = (32 if mode_id == "PRIMARY_B32" else 1), []
    with torch.inference_mode():
        for batch_index, group in enumerate(length_bucketed(rows, batch_size)):
            if time.monotonic() >= deadline:
                raise RuntimeError("RUNTIME_CAP_REACHED")
            packed = left_pad_teacher_forced(group, pad_token_id=13)
            logits = model(input_ids=torch.tensor(packed["input_ids"], device="cuda:0"), attention_mask=torch.tensor(packed["attention_mask"], device="cuda:0"), position_ids=torch.tensor(packed["position_ids"], device="cuda:0"), use_cache=False).logits
            for row_index, (row, boundary) in enumerate(zip(group, packed["boundaries"])):
                positions = extract_target_logit_indices(boundary)
                if len(positions) != len(row["target_ids"]):
                    raise RuntimeError("TARGET_LOGIT_LENGTH_MISMATCH")
                # Do not materialize BxSxV logits on CPU.  Select only causal
                # target positions on-device and transfer scalar rank/margins.
                selected = logits[row_index].index_select(0, torch.tensor(positions, device=logits.device)).float()
                targets = torch.tensor(row["target_ids"], device=selected.device, dtype=torch.long)
                correct = selected.gather(1, targets.unsqueeze(1)).squeeze(1)
                ranks = (selected > correct.unsqueeze(1)).sum(dim=1).add(1)
                incorrect = selected.clone()
                incorrect.scatter_(1, targets.unsqueeze(1), float("-inf"))
                margins = correct - incorrect.max(dim=1).values
                rank_values, margin_values = ranks.detach().cpu().tolist(), margins.detach().cpu().tolist()
                components = {name: {"token_count": 0, "correct_top1_count": 0, "correct_top2_count": 0, "margin_sum": 0.0} for name in COMPONENTS}
                error_index, parser_status = first_free_running_error(row, condition)
                error = None
                for target_index, (rank, margin) in enumerate(zip(rank_values, margin_values)):
                    component = component_name(row, target_index)
                    bucket = components[component]; bucket["token_count"] += 1; bucket["correct_top1_count"] += int(rank == 1); bucket["correct_top2_count"] += int(rank <= 2); bucket["margin_sum"] += float(margin)
                    if target_index == error_index:
                        error = {"target_index": target_index, "component": component, "rank": int(rank), "margin": float(margin), "parser_status": parser_status}
                results.append({"checkpoint_condition": condition, "mode_id": mode_id, "episode_id": row["episode_id"], "family": row["family"], "prompt_sha256": row["prompt_sha256"], "effective_batch_size": len(group), "batch_index": batch_index, "components": components, "first_free_running_error": error, "free_running_parser_status": parser_status, "target_token_count": len(row["target_ids"]), "target_token_ids_persisted": False})
    return results


def runtime_preflight(binding: dict) -> tuple[dict, dict, dict, dict]:
    paths = {key: Path(value) for key, value in binding["input_paths"].items()}
    required = {"raw_predictions", "input_manifest", "sealed_sidecar", "v7_manifest", "family_balanced_manifest"}
    if required - set(paths):
        raise RuntimeError("BINDING_INPUT_PATHS_INCOMPLETE")
    for key, path in paths.items():
        if sha(path) != binding["expected_hashes"].get(key):
            raise RuntimeError(f"RUNTIME_INPUT_SHA_MISMATCH:{key}")
    return select_raw_pairs(paths["raw_predictions"]), load_json(paths["input_manifest"]), load_json(paths["sealed_sidecar"]), {"RECONSTRUCTED_FOUNDATION_V2_V7": verify_manifest(paths["v7_manifest"], binding["expected_hashes"]["v7_manifest"]), "FAMILY_BALANCED": verify_manifest(paths["family_balanced_manifest"], binding["expected_hashes"]["family_balanced_manifest"])}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--runtime-seconds", type=int, default=CAP_SECONDS); args = parser.parse_args()
    if not args.binding.is_file():
        raise RuntimeError("LAUNCH_BINDING_MISSING")
    binding = load_json(args.binding)
    required = {"protocol_id", "runtime_cap_seconds", "expected_hashes", "input_paths", "checkpoints"}
    if not required.issubset(binding) or binding["protocol_id"] != PROTOCOL:
        raise RuntimeError("LAUNCH_BINDING_SCHEMA_INVALID")
    paths = {key: Path(value) for key, value in binding["input_paths"].items()}
    live_lock = args.output.parent / f".{binding.get('nonce', 'missing')}.live.lock"
    lock_held = False
    try:
        require_launch(output=args.output, receipt=args.receipt, cap_seconds=args.runtime_seconds, expected_hashes=binding["expected_hashes"], actual_paths=paths, live_lock=live_lock)
        lock_held = True
        if binding["runtime_cap_seconds"] != CAP_SECONDS or args.runtime_seconds != CAP_SECONDS:
            raise RuntimeError("BINDING_RUNTIME_CAP_DRIFT")
        raw_pairs, input_manifest, sidecar, manifests = runtime_preflight(binding)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA_BF16_UNAVAILABLE")
        deadline, started, results = time.monotonic() + CAP_SECONDS, time.monotonic(), []
        for condition in CONDITIONS:
            manifest = manifests[condition]; tokenizer = AutoTokenizer.from_pretrained(manifest["base_path"], local_files_only=True); tokenizer.pad_token_id = 13; tokenizer.padding_side = "left"
            if tokenizer.padding_side != "left":
                raise RuntimeError("LEFT_PADDING_UNAVAILABLE")
            rows = runtime_rows(input_manifest, sidecar, raw_pairs, tokenizer)
            model = AutoModelForCausalLM.from_pretrained(manifest["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
            model = PeftModel.from_pretrained(model, manifest["adapter_path"], is_trainable=False).eval()
            try:
                results.extend(score_batches(model, rows, torch, condition, "PRIMARY_B32", deadline))
                subset = set(binding.get("batch1_subset_episode_ids", []))
                if len(subset) != 12:
                    raise RuntimeError("BATCH1_SUBSET_BINDING_INVALID")
                results.extend(score_batches(model, [row for row in rows if row["episode_id"] in subset], torch, condition, "SENSITIVITY_B1", deadline))
            finally:
                del model; torch.cuda.empty_cache()
        require_complete_paired_rows([row for row in results if row["mode_id"] == "PRIMARY_B32"])
        if len([row for row in results if row["mode_id"] == "SENSITIVITY_B1"]) != 24:
            raise RuntimeError("BATCH1_RESULT_COMPLETENESS_INVALID")
        args.output.mkdir(parents=True, exist_ok=False); raw_path = args.output / "RAW_RANK_MARGIN.jsonl"
        raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\\n" for row in results), encoding="utf-8", newline="\\n")
        atomic(args.receipt, {"protocol_id": PROTOCOL, "status": "COMPLETE_NO_UPDATE", "raw_sha256": sha(raw_path), "raw_rows": len(results), "runtime_seconds": time.monotonic() - started, "optimizer_steps": 0, "training": False, "backward": False, "generation": False, "final_audit_opened": False, "target_token_ids_persisted": False})
        return 0
    except Exception as exc:
        if not args.receipt.exists():
            failure_receipt(receipt=args.receipt, reason=str(exc))
        raise
    finally:
        if lock_held:
            release_live_lock(live_lock)


if __name__ == "__main__":
    raise SystemExit(main())
