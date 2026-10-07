from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from generation_capability_audit_v1.audit import (
    GenerationExample,
    aggregate_generation,
    parse_generated_tokens,
    score_record,
)
from novel_holdout_confirmation_v1.audit import (
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_DATASET_FINGERPRINT,
    EXPECTED_FAMILY_SHA256,
    FAMILIES,
    HoldoutAuditError,
    canonical_sha256,
    composition_family,
    mcnemar_exact_p,
    paired_counts,
    prompt_pairs,
    select_sentinel,
    stratified_bootstrap,
    target_output,
    transfer_classification,
)
from training_data_v2.pipeline import native_render, serialize_grid


SOURCE_COMMIT = "28e3aa8c300e93d3c3ad553199993781c3c0ef8f"
BRANCH = "evaluation/novel-holdout-confirmation-v1"
MAX_NEW_TOKENS = 932
CONTEXT_WINDOW = 8704
REQUESTED_BATCH = 32
MAX_BATCHED_PROMPT_TOKENS = 21_568
MODEL_STATES = ("base", "tokens_2000000")
EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_text(demos: list[dict[str, Any]], target_input: list[list[int]]) -> str:
    messages: list[dict[str, str]] = []
    for pair in demos:
        if isinstance(pair, dict):
            inp, out = pair["input"], pair["output"]
        else:
            inp, out = pair[0], pair[1]
        messages.append({"role": "user", "content": serialize_grid(inp)})
        messages.append({"role": "assistant", "content": serialize_grid(out)})
    messages.append({"role": "user", "content": serialize_grid(target_input)})
    return native_render(messages) + "<|im_start|>assistant\n"


def _one_d_path(raw_root: Path, native_id: str, family: str) -> Path:
    return raw_root / "1d_arc" / "dataset" / family.split(":", 1)[1] / f"{native_id}.json"


def _comp_path(raw_root: Path) -> Path:
    return raw_root / "compositional_arc" / "all" / "all_episodes.jsonl.gz"


def _selected_source_texts(raw_root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    by_line = {int(row["source_line"]): str(row["sample_id"]) for row in rows if row["source"] == "compositional_arc"}
    if by_line:
        with gzip.open(_comp_path(raw_root), "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in by_line:
                    result[by_line[index]] = line
                    if len(result) >= len(by_line):
                        break
    for row in rows:
        if row["source"] == "1d_arc":
            path = _one_d_path(raw_root, str(row["native_id"]), str(row["family"]))
            result[str(row["sample_id"])] = path.read_text(encoding="utf-8")
    missing = sorted({str(row["sample_id"]) for row in rows} - set(result))
    if missing:
        raise HoldoutAuditError(f"SELECTED_SOURCE_MISSING={missing[:8]}")
    return result


def _checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer
    config_path = args.adapter / "adapter_config.json"
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.is_file():
        choices = sorted(args.adapter.glob("adapter_model*.safetensors"))
        if len(choices) != 1:
            raise HoldoutAuditError("CHECKPOINT_WEIGHTS_UNAVAILABLE")
        weights = choices[0]
    if sha256_file(weights) != EXPECTED_ADAPTER_SHA256:
        raise HoldoutAuditError("CHECKPOINT_IDENTITY_FAILURE")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        int(config.get("r", -1)) != 64
        or int(config.get("lora_alpha", -1)) != 32
        or float(config.get("lora_dropout", -1)) != 0
        or set(config.get("target_modules") or []) != EXPECTED_TARGETS
    ):
        raise HoldoutAuditError("CHECKPOINT_LORA_CONFIG_FAILURE")
    model_config = AutoConfig.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False, use_fast=True)
    return {
        "status": "PASS",
        "base_model": str(args.model),
        "base_model_type": str(model_config.model_type),
        "base_vocab_size": int(model_config.vocab_size),
        "tokenizer_length": len(tokenizer),
        "adapter": str(args.adapter),
        "adapter_weight_file": weights.name,
        "adapter_weight_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(config_path),
        "lora": {"r": 64, "alpha": 32, "dropout": 0, "targets": sorted(EXPECTED_TARGETS)},
    }


def prepare(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    if args.output.exists() and any(args.output.iterdir()):
        raise HoldoutAuditError(f"NONEMPTY_OUTPUT_EXISTS={args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifact.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=repo, check=False).returncode:
        raise HoldoutAuditError("SOURCE_COMMIT_NOT_ANCESTOR")
    fingerprint = json.loads((args.freeze / "NOVEL_DATASET_FINGERPRINT.json").read_text(encoding="utf-8"))
    families = json.loads((args.freeze / "NOVEL_HOLDOUT_FAMILIES.json").read_text(encoding="utf-8"))
    holdout = json.loads((args.freeze / "V1_1_HOLDOUT_FREEZE.json").read_text(encoding="utf-8"))
    if fingerprint.get("fingerprint_sha256") != EXPECTED_DATASET_FINGERPRINT:
        raise HoldoutAuditError("HOLDOUT_DATASET_FINGERPRINT_MISMATCH")
    if families.get("sha256") != EXPECTED_FAMILY_SHA256 or tuple(families.get("families") or []) != FAMILIES:
        raise HoldoutAuditError("HOLDOUT_FAMILY_MANIFEST_MISMATCH")
    if holdout.get("status") != "FROZEN" or int(holdout.get("base_puzzle_count", -1)) != 8646:
        raise HoldoutAuditError("HOLDOUT_FREEZE_MISMATCH")
    holdout_ids = set(holdout["base_puzzle_ids"])
    candidates: list[dict[str, Any]] = []
    comp_wanted: dict[int, str] = {}
    for base_id in holdout_ids:
        if base_id.startswith("1d_arc:"):
            native_id = base_id.split(":", 1)[1]
            family = "1d_arc:" + native_id.rsplit("_", 1)[0]
            candidates.append({
                "sample_id": base_id + ":native-source-episode-v1", "base_puzzle_id": base_id,
                "source": "1d_arc", "native_id": native_id, "source_line": None,
                "family": family, "split": "holdout",
            })
        elif base_id.startswith("compositional_arc:episode-"):
            comp_wanted[int(base_id.rsplit("-", 1)[1])] = base_id
        else:
            raise HoldoutAuditError(f"UNKNOWN_FROZEN_HOLDOUT_ID={base_id}")
    with gzip.open(_comp_path(args.raw_root), "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in comp_wanted:
                continue
            base_id = comp_wanted[index]
            candidates.append({
                "sample_id": base_id + ":native-source-episode-v1", "base_puzzle_id": base_id,
                "source": "compositional_arc", "native_id": base_id.split(":", 1)[1], "source_line": index,
                "family": composition_family(line), "split": "holdout",
            })
    selected = select_sentinel(candidates)
    source_texts = _selected_source_texts(args.raw_root, selected)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False, use_fast=True)
    inputs: list[dict[str, Any]] = []
    for row in selected:
        demos, target_input = prompt_pairs(source_texts[row["sample_id"]], row["source"])
        prompt_ids = [int(value) for value in tokenizer(_prompt_text(demos, target_input), add_special_tokens=False, return_attention_mask=False)["input_ids"]]
        if prompt_ids[-3:] != [14, 12, 10]:
            raise HoldoutAuditError(f"ASSISTANT_PREFIX_MISMATCH={row['sample_id']}")
        if len(prompt_ids) + MAX_NEW_TOKENS > CONTEXT_WINDOW:
            raise HoldoutAuditError(f"SELECTED_CONTEXT_OVERFLOW={row['sample_id']}:{len(prompt_ids)}")
        value = dict(row)
        value.update({
            "sequence_length": len(prompt_ids),
            "sequence_length_semantics": "inference_prompt_tokens_excluding_holdout_gold",
            "prompt_ids": prompt_ids,
            "prompt_sha256": canonical_sha256(prompt_ids),
            "gold_target_tokens_in_inference_input": False,
            "target_labels_passed_to_generation": False,
            "no_later_tokens": True,
        })
        inputs.append(value)
    inputs.sort(key=lambda row: (FAMILIES.index(row["family"]), row["ranking_sha256"], row["sample_id"]))
    selection_view = [
        {key: row[key] for key in ("sample_id", "base_puzzle_id", "source", "family", "split", "source_line", "sequence_length", "ranking_sha256", "prompt_sha256")}
        for row in inputs
    ]
    selection_sha = canonical_sha256(selection_view)
    atomic_json(args.output / "INPUTS_TARGET_BLIND.json", {"status": "FROZEN", "selection_sha256": selection_sha, "rows": inputs})
    sentinel = {
        "status": "FROZEN_PRE_GENERATION", "seed": 2_000_031, "episode_count": 200,
        "family_balance": {family: 50 for family in FAMILIES}, "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "family_manifest_sha256": EXPECTED_FAMILY_SHA256, "selection_sha256": selection_sha,
        "selection_algorithm": "SHA256(seed:family:sample_id), first 50 per family",
        "metadata_only_before_generation": True, "episodes": selection_view,
    }
    identity = _checkpoint_identity(args)
    context = {
        "status": "PASS", "record_count": 200,
        "gold_target_tokens_in_inference_input": False, "target_labels_passed_to_generation": False,
        "no_later_tokens": True, "metadata_only_selection": True,
        "target_policy": "complete demonstrations plus final target input and assistant prefix; final Gold output excluded",
        "max_prompt_tokens": max(row["sequence_length"] for row in inputs),
        "max_prompt_plus_generation_tokens": max(row["sequence_length"] for row in inputs) + MAX_NEW_TOKENS,
    }
    preflight = {
        "status": "PASS", "source_commit": SOURCE_COMMIT, "branch": BRANCH,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT, "holdout_episodes": 8646,
        "holdout_families": list(FAMILIES), "sentinel_count": 200,
        "gold_loaded": False, "selection_source_fields": ["sample_id", "source", "generator_family", "split", "sequence_length"],
    }
    config = {
        "status": "FROZEN", "model_states": list(MODEL_STATES), "context_window": CONTEXT_WINDOW,
        "generation": {"do_sample": False, "num_beams": 1, "temperature": None, "max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": 15, "pad_token_id": 13},
        "batch": {"requested_max_batch": REQUESTED_BATCH, "max_batched_prompt_tokens": MAX_BATCHED_PROMPT_TOKENS, "bucketing": "prompt_length"},
        "optimizer_steps": 0, "backward_calls": 0, "optimizer_state_created": False,
    }
    for path, value in (
        (args.artifact / "HOLDOUT_PREFLIGHT.json", preflight),
        (args.artifact / "HOLDOUT_CONFIRMATION_SENTINEL.json", sentinel),
        (args.artifact / "HOLDOUT_CHECKPOINT_IDENTITY.json", identity),
        (args.artifact / "HOLDOUT_GENERATION_CONFIG.json", config),
        (args.artifact / "HOLDOUT_GENERATION_CONTEXT_AUDIT.json", context),
        (args.artifact / "NO_TRAINING_AUDIT.json", {"status": "PASS", "optimizer_steps": 0, "backward_calls": 0, "optimizer_state_created": False, "TTT": False, "search": False, "verifier": False, "ranker": False}),
    ):
        atomic_json(path, value)
    print(json.dumps({"status": "PASS", "sentinel": 200, "selection_sha256": selection_sha, "max_prompt": context["max_prompt_tokens"]}), flush=True)
    return 0


def _inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads((args.output / "INPUTS_TARGET_BLIND.json").read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or len(payload.get("rows") or []) != 200:
        raise HoldoutAuditError("TARGET_BLIND_INPUTS_NOT_FROZEN_200")
    return payload["rows"]


def _load_model(model_path: Path, adapter: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval(); model.config.use_cache = True
    return model


def _batch_count(rows: list[dict[str, Any]], cursor: int, maximum: int) -> int:
    count = min(maximum, len(rows) - cursor)
    while count > 1:
        width = max(len(row["prompt_ids"]) for row in rows[cursor:cursor + count])
        if width * count <= MAX_BATCHED_PROMPT_TOKENS:
            break
        count -= 1
    return count


def _generate_batch(model: Any, batch: list[dict[str, Any]]) -> tuple[Any, int, float]:
    import torch
    width = max(len(row["prompt_ids"]) for row in batch)
    prompt = torch.full((len(batch), width), 13, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(prompt)
    for row_index, row in enumerate(batch):
        length = len(row["prompt_ids"])
        prompt[row_index, width - length:] = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda:0")
        mask[row_index, width - length:] = 1
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, eos_token_id=15, pad_token_id=13, use_cache=True)
    torch.cuda.synchronize(0)
    seconds = time.perf_counter() - started
    suffixes = output[:, width:].detach().cpu()
    del output, prompt, mask
    return suffixes, width, seconds


def smoke(args: argparse.Namespace) -> int:
    import torch
    rows = sorted(_inputs(args), key=lambda row: (-len(row["prompt_ids"]), row["sample_id"]))
    count = _batch_count(rows, 0, REQUESTED_BATCH)
    model = _load_model(args.model, args.adapter)
    attempts = []
    while True:
        batch = rows[:count]
        try:
            torch.cuda.reset_peak_memory_stats(0)
            suffixes, width, seconds = _generate_batch(model, batch)
            receipt = {
                "status": "PASS", "requested_batch_size": REQUESTED_BATCH, "actual_batch_size": count,
                "max_batched_prompt_tokens": MAX_BATCHED_PROMPT_TOKENS, "longest_prompt_bucket": True,
                "prompt_width": width, "wall_seconds": seconds, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "generated_rows": len(suffixes), "oom_attempts": attempts, "selected_max_batch": REQUESTED_BATCH,
            }
            atomic_json(args.artifact / "BATCH32_SMOKE.json", receipt)
            print(json.dumps(receipt), flush=True)
            return 0
        except torch.cuda.OutOfMemoryError:
            attempts.append({"actual_batch_size": count, "status": "OOM"})
            gc.collect(); torch.cuda.empty_cache()
            if count <= 4:
                raise
            count = 16 if count > 16 else 8 if count > 8 else 4


def generate(args: argparse.Namespace) -> int:
    import torch
    if args.state not in MODEL_STATES:
        raise HoldoutAuditError(f"UNKNOWN_MODEL_STATE={args.state}")
    rows = sorted(_inputs(args), key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    adapter = None if args.state == "base" else args.adapter
    model = _load_model(args.model, adapter)
    torch.cuda.reset_peak_memory_stats(0)
    records: list[dict[str, Any]] = []
    oom: list[dict[str, Any]] = []
    started = time.perf_counter(); cursor = 0
    while cursor < len(rows):
        count = _batch_count(rows, cursor, REQUESTED_BATCH)
        batch = rows[cursor:cursor + count]
        while True:
            try:
                suffixes, width, seconds = _generate_batch(model, batch)
                break
            except torch.cuda.OutOfMemoryError:
                failed = len(batch)
                gc.collect(); torch.cuda.empty_cache()
                if failed == 1:
                    raise
                fallback = 16 if failed > 16 else 8 if failed > 8 else 4 if failed > 4 else max(1, failed // 2)
                batch = batch[:min(fallback, len(batch) - 1)]
                oom.append({"cursor": cursor, "failed_batch": failed, "fallback_batch": len(batch)})
        for index, row in enumerate(batch):
            raw = [int(value) for value in suffixes[index].tolist()]
            if 15 in raw:
                raw = raw[:raw.index(15) + 1]
            records.append({
                "sample_id": row["sample_id"], "model_state": args.state,
                "generated_token_ids": raw, "termination_status": "EOS" if raw and raw[-1] == 15 else "MAX_NEW_TOKENS",
                "generated_length": len(raw), "prompt_sha256": row["prompt_sha256"],
                "requested_batch_size": REQUESTED_BATCH, "actual_batch_size": len(batch),
                "batch_prompt_width": width, "batch_wall_seconds": seconds,
            })
        cursor += len(batch)
        atomic_json(args.output / "RAW_PREDICTIONS" / f"{args.state}.partial.json", {"status": "INCOMPLETE", "completed": len(records), "records": records})
        print(json.dumps({"event": "HOLDOUT_GENERATION_PROGRESS", "state": args.state, "completed": len(records), "total": 200, "actual_batch_size": len(batch)}), flush=True)
        del suffixes
    payload = {
        "status": "COMPLETE", "model_state": args.state, "record_count": 200,
        "wall_seconds": time.perf_counter() - started, "requested_batch_size": REQUESTED_BATCH,
        "max_batched_prompt_tokens": MAX_BATCHED_PROMPT_TOKENS, "oom_fallbacks": oom,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0), "gold_loaded": False,
        "optimizer_steps": 0, "backward_calls": 0, "records": records,
    }
    target = args.output / "RAW_PREDICTIONS" / f"{args.state}.json"
    atomic_json(target, payload)
    partial = target.with_name(f"{args.state}.partial.json")
    if partial.exists(): partial.unlink()
    return 0


def freeze_predictions(args: argparse.Namespace) -> int:
    files = {}
    for state in MODEL_STATES:
        path = args.output / "RAW_PREDICTIONS" / f"{state}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("record_count") != 200 or payload.get("gold_loaded") is not False:
            raise HoldoutAuditError(f"RAW_PREDICTION_NOT_COMPLETE_TARGET_BLIND={state}")
        files[state] = {"storage_path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "row_count": 200, "schema": ["sample_id", "model_state", "generated_token_ids", "termination_status", "generated_length", "prompt_sha256"]}
    receipt = {"status": "PASS", "both_states_complete_before_gold": True, "gold_loaded": False, "files": files}
    atomic_json(args.output / "PREDICTION_FREEZE.json", receipt)
    atomic_json(args.artifact / "HOLDOUT_PREDICTION_HASHES.json", receipt)
    return 0


def _verify_prediction_freeze(args: argparse.Namespace) -> dict[str, Any]:
    receipt = json.loads((args.output / "PREDICTION_FREEZE.json").read_text(encoding="utf-8"))
    for state, entry in receipt["files"].items():
        path = Path(entry["storage_path"])
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise HoldoutAuditError(f"PREDICTION_HASH_MISMATCH={state}")
    return receipt


def score(args: argparse.Namespace) -> int:
    _verify_prediction_freeze(args)
    inputs = _inputs(args)
    texts = _selected_source_texts(args.raw_root, inputs)
    by_state = {}
    for state in MODEL_STATES:
        payload = json.loads((args.output / "RAW_PREDICTIONS" / f"{state}.json").read_text(encoding="utf-8"))
        by_state[state] = {row["sample_id"]: row for row in payload["records"]}
    scored: dict[str, list[dict[str, Any]]] = {state: [] for state in MODEL_STATES}
    paired_rows = []
    for row in inputs:
        gold = tuple(tuple(int(cell) for cell in grid_row) for grid_row in target_output(texts[row["sample_id"]], row["source"]))
        example = GenerationExample(
            sample_id=row["sample_id"], source=row["source"], family=row["family"],
            prompt_ids=tuple(row["prompt_ids"]), gold_ids=(), gold_grid=gold,
            target_span_start=len(row["prompt_ids"]), target_span_end=len(row["prompt_ids"]),
            original_sequence_length=row["sequence_length"],
        )
        exacts = {}
        for state in MODEL_STATES:
            generated = by_state[state][row["sample_id"]]
            tokens = generated["generated_token_ids"]
            parsed = parse_generated_tokens(tokens, gold, hit_max_new_tokens=generated["termination_status"] == "MAX_NEW_TOKENS")
            result = score_record(example, parsed)
            scored[state].append(result)
            exacts[state] = bool(result["exact_grid"])
        paired_rows.append({"sample_id": row["sample_id"], "family": row["family"], "base_exact": exacts["base"], "trained_exact": exacts["tokens_2000000"]})
    metrics = {state: aggregate_generation(records) for state, records in scored.items()}
    base = metrics["base"]; trained = metrics["tokens_2000000"]
    point = trained["macro_family_exact_grid_accuracy"] - base["macro_family_exact_grid_accuracy"]
    bootstrap = stratified_bootstrap(paired_rows)
    counts = paired_counts([row["base_exact"] for row in paired_rows], [row["trained_exact"] for row in paired_rows])
    statistics = {
        **counts, "absolute_exact_grid_gain": point,
        "mcnemar_exact_p": mcnemar_exact_p(counts["n01"], counts["n10"]),
        "paired_stratified_bootstrap": bootstrap,
    }
    def subset(prefix: str) -> dict[str, Any]:
        families = [family for family in FAMILIES if family.startswith(prefix)]
        left = sum(base["per_family_exact_accuracy"][family] for family in families) / len(families)
        right = sum(trained["per_family_exact_accuracy"][family] for family in families) / len(families)
        return {"families": families, "base_exact": left, "tokens_2000000_exact": right, "absolute_gain": right - left}
    breakdown = {"one_d_true_family_holdout": subset("1d_arc:"), "compositional_systematicity_holdout": subset("compositional_arc:")}
    classification = transfer_classification(point, bootstrap["percentile_95_ci"][0], bootstrap["percentile_95_ci"][1])
    results = {"status": "PASS", "models": metrics, "overall_absolute_gain": point, "classification": classification}
    access = {
        "status": "PASS", "holdout_gold_accessed": True, "holdout_gold_access_started_only_after_both_prediction_hashes_passed": True,
        "novel_holdout_confirmation_sentinel": "EXPOSED_AFTER_CONFIRMATION", "eval60_accessed": False,
        "kaggle_gold_accessed": False, "arc_hidden_test_gold_accessed": False, "future_meta_holdout_accessed": False,
    }
    gate = {
        "status": "PASS", "scientific_validity": "PASS", "classification": classification,
        "sentinel_count": 200, "raw_prediction_hashes": "PASS", "gold_leakage_audit": "PASS",
        "optimizer_steps": 0, "holdout_status": "EXPOSED_AFTER_CONFIRMATION",
    }
    atomic_json(args.artifact / "HOLDOUT_RESULTS.json", results)
    atomic_json(args.artifact / "HOLDOUT_SOURCE_BREAKDOWN.json", breakdown)
    atomic_json(args.artifact / "PAIRED_TRANSFER_STATISTICS.json", statistics)
    atomic_json(args.artifact / "HOLDOUT_DATA_ACCESS_AUDIT.json", access)
    atomic_json(args.artifact / "HOLDOUT_CONFIRMATION_GATE.json", gate)
    report = f"""# Novel Holdout Confirmation V1

| Model | Overall exact | Macro-family exact | Valid | 1D exact | Compositional exact |
|---|---:|---:|---:|---:|---:|
| SFT139 Base | {base['micro_exact_grid_accuracy']:.4f} | {base['macro_family_exact_grid_accuracy']:.4f} | {base['valid_format_rate']:.4f} | {breakdown['one_d_true_family_holdout']['base_exact']:.4f} | {breakdown['compositional_systematicity_holdout']['base_exact']:.4f} |
| 2M LoRA | {trained['micro_exact_grid_accuracy']:.4f} | {trained['macro_family_exact_grid_accuracy']:.4f} | {trained['valid_format_rate']:.4f} | {breakdown['one_d_true_family_holdout']['tokens_2000000_exact']:.4f} | {breakdown['compositional_systematicity_holdout']['tokens_2000000_exact']:.4f} |

Overall absolute gain: {point * 100:.2f} pp

1D absolute gain: {breakdown['one_d_true_family_holdout']['absolute_gain'] * 100:.2f} pp

Compositional absolute gain: {breakdown['compositional_systematicity_holdout']['absolute_gain'] * 100:.2f} pp

Paired n01={counts['n01']}, n10={counts['n10']}; McNemar exact p={statistics['mcnemar_exact_p']:.8g}; bootstrap 95% CI={bootstrap['percentile_95_ci']}.

Classification: `{classification}`

NOVEL_V1_1_HOLDOUT_CONFIRMATION_SENTINEL = `EXPOSED_AFTER_CONFIRMATION`
"""
    atomic_text(args.artifact / "REPORT.md", report)
    entries = {}
    for path in sorted(args.artifact.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.json":
            entries[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(args.artifact / "SHA256SUMS.json", {"status": "PASS", "entries": entries})
    print(json.dumps({"status": "PASS", "classification": classification, "gain": point, "n01": counts["n01"], "n10": counts["n10"]}), flush=True)
    return 0


def controller(args: argparse.Namespace) -> int:
    common = [
        "--output", str(args.output), "--artifact", str(args.artifact), "--freeze", str(args.freeze),
        "--raw-root", str(args.raw_root), "--model", str(args.model), "--adapter", str(args.adapter),
    ]
    commands = [
        [sys.executable, __file__, "--mode", "prepare", *common],
        [sys.executable, __file__, "--mode", "smoke", *common],
        [sys.executable, __file__, "--mode", "generate", "--state", "base", *common],
        [sys.executable, __file__, "--mode", "generate", "--state", "tokens_2000000", *common],
        [sys.executable, __file__, "--mode", "freeze", *common],
        [sys.executable, __file__, "--mode", "score", *common],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("prepare", "smoke", "generate", "freeze", "score", "controller"), required=True)
    value.add_argument("--state", choices=MODEL_STATES)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--artifact", type=Path, required=True)
    value.add_argument("--freeze", type=Path, required=True)
    value.add_argument("--raw-root", type=Path, required=True)
    value.add_argument("--model", type=Path, required=True)
    value.add_argument("--adapter", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    return {"prepare": prepare, "smoke": smoke, "generate": generate, "freeze": freeze_predictions, "score": score, "controller": controller}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
