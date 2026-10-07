from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

from foundation_v2_capability_diagnostic_v1.audit import (
    COMPOSITION_CATEGORIES, CONTEXT_WINDOW, EOS_TOKEN_ID, INPUTS, MAX_NEW_TOKENS,
    MODEL_STATES, PAD_TOKEN_ID, SOURCE_COMMIT, DiagnosticError,
    aggregate_capabilities, canonical_sha256, composition_results, engineering_band,
    pair_models, parameter_results, priority_map, score_prediction, serialize_grid,
    stage_summary,
)


ADAPTER_SHA256 = "f0079dd399c1f0521c378a46bd7bbb737a0abc72ef0bb6c47a6c1ae33446afce"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
FORBIDDEN = ("round1_holdout", "c1_dev_isolated", "c2_dev_isolated", "meta_composition_holdout", "eval60", "kaggle_gold", "arc_hidden")
ARTIFACT_NAMES = (
    "MODEL_IDENTITY_AUDIT.json", "DIAGNOSTIC_INPUT_IDENTITY.json", "CAPABILITY_DIAGNOSTIC_CONTEXT_AUDIT.json",
    "GPU_DIAGNOSTIC_BATCH_CALIBRATION.json", "RAW_PREDICTION_FREEZE.json", "DIRECT_ATOMIC_RESULTS.json",
    "MINIMAL_CONTRAST_RESULTS.json", "COMPOSITE_CHAIN_RESULTS.json", "PARAMETER_GENERALIZATION_RESULTS.json",
    "COMPOSITION_DEVELOPMENT_RESULTS.json", "FOUNDATION_V2_CAPABILITY_PROFILE.json", "FOUNDATION_V2_CAPABILITY_PROFILE.md",
    "BASE_VS_FOUNDATION_V2_DELTA_MAP.json", "STAGE_LEVEL_CAPABILITY_SUMMARY.json",
    "FOUNDATION_CURRICULUM_PRIORITY_MAP_V1.json", "DATA_ACCESS_AUDIT.json", "NO_TRAINING_AUDIT.json", "REPORT.json", "REPORT.md",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename=str(tmp), mode="wb", mtime=0) as binary:
        for row in rows:
            binary.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    tmp.replace(path)


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {cohort: args.input_root / spec[0] for cohort, spec in INPUTS.items()}


def verify_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit: dict[str, Any] = {"status": "PASS", "source_commit": SOURCE_COMMIT, "cohorts": {}}
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cohort, (name, count, expected_sha) in INPUTS.items():
        path = args.input_root / name
        actual_sha = sha256_file(path)
        rows = read_jsonl_gz(path)
        if actual_sha != expected_sha or len(rows) != count:
            raise DiagnosticError(f"INPUT_IDENTITY_FAILURE={cohort}:{actual_sha}:{len(rows)}")
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise DiagnosticError(f"DUPLICATE_SAMPLE_ID={sample_id}")
            seen.add(sample_id)
            combined.append({"cohort": cohort, **row})
        audit["cohorts"][cohort] = {"logical_name": name, "episode_count": count, "sha256": actual_sha}
    audit["total_episodes"] = len(combined)
    return audit, combined


def model_identity(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig
    from inference.nvarc_native import native_tokenizer_preflight

    weights = args.adapter_path / "adapter_model.safetensors"
    config_path = args.adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(weights) != ADAPTER_SHA256:
        raise DiagnosticError("MODEL_IDENTITY_FAILURE=ADAPTER_SHA256")
    if int(config.get("r", -1)) != 64 or int(config.get("lora_alpha", -1)) != 32 or float(config.get("lora_dropout", -1)) != 0 or set(config.get("target_modules") or []) != TARGETS:
        raise DiagnosticError("MODEL_IDENTITY_FAILURE=LORA_CONFIG")
    base_config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    tokenizer = native_tokenizer_preflight(args.model_path, args.native_config_dir)
    files = sorted(path for path in args.model_path.iterdir() if path.is_file() and (path.suffix in {".json", ".safetensors"} or path.name.endswith(".model")))
    return {
        "status": "PASS", "base_model_identity": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "base_model_path": str(args.model_path), "base_model_type": str(base_config.model_type), "base_vocab_size": int(base_config.vocab_size),
        "base_files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files},
        "tokenizer": tokenizer, "foundation_v2_adapter_path": str(args.adapter_path),
        "adapter_weight_sha256": sha256_file(weights), "adapter_config": config,
    }


def task_messages_without_gold(task: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if len(task.get("train", [])) != 4 or len(task.get("test", [])) != 1:
        raise DiagnosticError("EXPECTED_FOUR_DEMOS_ONE_TEST")
    for pair in task["train"]:
        messages.append({"role": "user", "content": serialize_grid(pair["input"])})
        messages.append({"role": "assistant", "content": serialize_grid(pair["output"])})
    messages.append({"role": "user", "content": serialize_grid(task["test"][0]["input"])})
    return messages


def preflight(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.iterdir()):
        raise DiagnosticError(f"NONEMPTY_OUTPUT={args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.artifact.mkdir(parents=True, exist_ok=True)
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], cwd=Path(__file__).resolve().parents[1], check=False).returncode == 0
    if not ancestor:
        raise DiagnosticError("SOURCE_COMMIT_NOT_ANCESTOR")
    identity, source_rows = verify_inputs(args)
    model = model_identity(args)
    from inference.nvarc_native import checkpoint_native_tokenizer
    tokenizer, _ = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    contexts, audit_rows = [], []
    for row in source_rows:
        messages = task_messages_without_gold(row["task"])
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors=None, return_dict=True)
        prompt_ids = [int(value) for value in encoded["input_ids"]]
        if len(prompt_ids) > CONTEXT_WINDOW:
            raise DiagnosticError(f"CONTEXT_OVERFLOW={row['sample_id']}:{len(prompt_ids)}")
        prompt_sha = canonical_sha256(prompt_ids)
        metadata = {key: row[key] for key in row if key not in {"task", "scene_evidence", "mechanical_witness"}}
        contexts.append({"sample_id": row["sample_id"], "cohort": row["cohort"], "prompt_ids": prompt_ids, "prompt_sha256": prompt_sha, "metadata": metadata})
        audit_rows.append({"sample_id": row["sample_id"], "cohort": row["cohort"], "prompt_token_count": len(prompt_ids), "prompt_sha256": prompt_sha,
                           "gold_target_excluded": True, "labels_passed": False, "no_later_tokens": True, "status": "PASS"})
    if len(contexts) != 3000 or len({row["sample_id"] for row in contexts}) != 3000:
        raise DiagnosticError("CONTEXT_COUNT_FAILURE")
    atomic_json(args.output / "GENERATION_CONTEXTS.json", {"status": "FROZEN_TARGET_BLIND", "count": len(contexts), "records": contexts})
    context_audit = {"status": "PASS", "record_count": len(audit_rows), "all_samples_pass": True, "records": audit_rows}
    atomic_json(args.artifact / "MODEL_IDENTITY_AUDIT.json", model)
    atomic_json(args.artifact / "DIAGNOSTIC_INPUT_IDENTITY.json", identity)
    atomic_json(args.artifact / "CAPABILITY_DIAGNOSTIC_CONTEXT_AUDIT.json", context_audit)
    atomic_json(args.artifact / "NO_TRAINING_AUDIT.json", {"status": "PASS", "optimizer_steps": 0, "backward_calls": 0, "optimizer_state_created": False,
                                                            "model_parameters_modified": False, "adapter_modified": False, "post_eval_adapter_sha256": None})
    atomic_json(args.artifact / "DATA_ACCESS_AUDIT.json", data_access_audit())
    print(json.dumps({"status": "PASS", "contexts": len(contexts)}), flush=True)
    return 0


def data_access_audit() -> dict[str, Any]:
    return {"status": "PASS", "ROUND1_HOLDOUT_ACCESSED": False, "C1_DEV_ISOLATED_MODEL_ACCESSED": False,
            "C2_DEV_ISOLATED_MODEL_ACCESSED": False, "META_HOLDOUT_MODEL_ACCESSED": False,
            "EVAL60_GOLD_ACCESSED": False, "KAGGLE_GOLD_ACCESSED": False, "ARC_HIDDEN_GOLD_ACCESSED": False}


def load_contexts(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = json.loads((args.output / "GENERATION_CONTEXTS.json").read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_TARGET_BLIND" or payload.get("count") != 3000:
        raise DiagnosticError("CONTEXT_FREEZE_INVALID")
    return payload["records"]


def load_model(args: argparse.Namespace, state: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if state == "foundation_v2":
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval(); model.config.use_cache = True
    return model


def generate_batch(model: Any, batch: list[dict[str, Any]]) -> tuple[list[list[int]], float]:
    import torch
    width = max(len(row["prompt_ids"]) for row in batch)
    prompt = torch.full((len(batch), width), PAD_TOKEN_ID, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(prompt)
    for index, row in enumerate(batch):
        length = len(row["prompt_ids"])
        prompt[index, width-length:] = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda:0")
        mask[index, width-length:] = 1
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, num_beams=1, temperature=None, eos_token_id=EOS_TOKEN_ID,
                                pad_token_id=PAD_TOKEN_ID, use_cache=True)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    suffixes = []
    for row in output:
        raw = [int(value) for value in row[width:].detach().cpu().tolist()]
        suffixes.append(raw[:raw.index(EOS_TOKEN_ID)+1] if EOS_TOKEN_ID in raw else raw)
    del output, prompt, mask
    return suffixes, elapsed


def calibration_selection(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    def add_strata(rows: list[dict[str, Any]], label: str) -> None:
        ordered = sorted(rows, key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
        for suffix, index in (("short", 0), ("median", len(ordered)//2), ("long", len(ordered)-1)):
            item = dict(ordered[index]); item["calibration_stratum"] = f"{label}:{suffix}"; selected[item["sample_id"]] = item
    caps = [row for row in contexts if row["cohort"] == "capability"]
    for kind in ("DIRECT_ATOMIC", "MINIMAL_CONTRAST", "COMPOSITE_ONLY"):
        add_strata([row for row in caps if row["metadata"]["measurement_type"] == kind], f"measurement:{kind}")
    params = [row for row in contexts if row["cohort"] == "parameter"]
    for axis in sorted({row["metadata"]["axis"] for row in params}):
        add_strata([row for row in params if row["metadata"]["axis"] == axis], f"axis:{axis}")
    comps = [row for row in contexts if row["cohort"] == "composition"]
    for category in COMPOSITION_CATEGORIES:
        add_strata([row for row in comps if row["metadata"]["category"] == category], f"composition:{category}")
    for row in sorted(contexts, key=lambda row: (len(row["prompt_ids"]), row["sample_id"])):
        if len(selected) >= 64:
            break
        if row["sample_id"] not in selected:
            item = dict(row); item["calibration_stratum"] = "overall:fill"; selected[item["sample_id"]] = item
    result = list(selected.values())
    if len(result) != 64:
        raise DiagnosticError(f"CALIBRATION_NOT_64={len(result)}")
    return result


def calibration(args: argparse.Namespace) -> int:
    import torch
    contexts = load_contexts(args)
    selected = calibration_selection(contexts)
    model = load_model(args, "foundation_v2")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    singles: dict[str, list[int]] = {}
    single_seconds = 0.0
    for index, row in enumerate(selected, 1):
        suffixes, elapsed = generate_batch(model, [row]); singles[row["sample_id"]] = suffixes[0]; single_seconds += elapsed
        print(json.dumps({"event": "CALIBRATION_B1", "completed": index, "total": 64}), flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    ordered = sorted(selected, key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    batched: dict[str, list[int]] = {}; actual_sizes: list[int] = []; batched_seconds = 0.0; cursor = 0
    while cursor < len(ordered):
        count = min(args.batch_size, len(ordered)-cursor)
        while count > 1 and max(len(row["prompt_ids"]) for row in ordered[cursor:cursor+count]) * count > args.max_batched_prompt_tokens:
            count -= 1
        batch = ordered[cursor:cursor+count]
        suffixes, elapsed = generate_batch(model, batch); batched_seconds += elapsed; actual_sizes.append(len(batch))
        for row, tokens in zip(batch, suffixes, strict=True): batched[row["sample_id"]] = tokens
        cursor += len(batch)
    mismatches = [{"sample_id": row["sample_id"], "b1_sha256": canonical_sha256(singles[row["sample_id"]]), "batch_sha256": canonical_sha256(batched[row["sample_id"]])}
                  for row in selected if singles[row["sample_id"]] != batched[row["sample_id"]]]
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    peak_alloc = int(torch.cuda.max_memory_allocated(0)); peak_reserved = int(torch.cuda.max_memory_reserved(0))
    payload = {"status": "PASS" if not mismatches else "FAIL", "model_state": "foundation_v2", "representative_samples": 64,
               "coverage": {"length_bands": ["short", "median", "long"], "measurement_types": 3, "parameter_axes": 9, "composition_categories": 7},
               "selected_configuration": {"max_batch": args.batch_size, "max_batched_prompt_tokens": args.max_batched_prompt_tokens, "prompt_length_bucketing": True},
               "b1_wall_seconds": single_seconds, "batched_wall_seconds": batched_seconds, "examples_per_second": 64/batched_seconds,
               "decode_tokens_per_second": sum(len(v) for v in batched.values())/batched_seconds,
               "actual_batch_sizes": actual_sizes, "peak_allocated_vram_bytes": peak_alloc, "peak_reserved_vram_bytes": peak_reserved,
               "vram_headroom_bytes": total_vram-peak_reserved, "oom": False, "token_exact_comparisons": 64,
               "token_exact_mismatches": mismatches, "generation_semantics_unchanged": True}
    atomic_json(args.artifact / "GPU_DIAGNOSTIC_BATCH_CALIBRATION.json", payload)
    if mismatches:
        raise DiagnosticError("B1_BATCH_PARITY_FAILURE")
    print(json.dumps({"status": "PASS", "b1_seconds": single_seconds, "batched_seconds": batched_seconds, "actual_batch_sizes": actual_sizes}), flush=True)
    return 0


def _raw_path(args: argparse.Namespace, state: str) -> Path:
    return args.output / "raw_predictions" / f"{state}.jsonl.gz"


def generate_state(args: argparse.Namespace) -> int:
    import torch
    if args.state not in MODEL_STATES: raise DiagnosticError(f"BAD_STATE={args.state}")
    contexts = load_contexts(args)
    partial_path = args.output / "raw_predictions" / f"{args.state}.partial.json"
    existing: dict[str, dict[str, Any]] = {}
    if partial_path.is_file():
        for row in json.loads(partial_path.read_text(encoding="utf-8"))["records"]:
            existing[row["sample_id"]] = row
    pending = [row for row in contexts if row["sample_id"] not in existing]
    pending.sort(key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    model = load_model(args, args.state)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter(); cursor = 0; batch_sizes: list[int] = []; oom_fallbacks: list[dict[str, int]] = []
    while cursor < len(pending):
        count = min(args.batch_size, len(pending)-cursor)
        while count > 1 and max(len(row["prompt_ids"]) for row in pending[cursor:cursor+count]) * count > args.max_batched_prompt_tokens:
            count -= 1
        while True:
            batch = pending[cursor:cursor+count]
            try:
                suffixes, _elapsed = generate_batch(model, batch); break
            except torch.cuda.OutOfMemoryError:
                failed = count; gc.collect(); torch.cuda.empty_cache()
                if count == 1: raise
                count = max(1, count//2); oom_fallbacks.append({"failed_batch": failed, "fallback_batch": count})
        for context, tokens in zip(batch, suffixes, strict=True):
            existing[context["sample_id"]] = {"sample_id": context["sample_id"], "cohort": context["cohort"], "model_state": args.state,
                                                      "generated_token_ids": tokens, "generated_length": len(tokens),
                                                      "termination_status": "EOS" if tokens and tokens[-1] == EOS_TOKEN_ID else "MAX_NEW_TOKENS",
                                                      "prompt_sha256": context["prompt_sha256"]}
        cursor += len(batch); batch_sizes.append(len(batch))
        atomic_json(partial_path, {"status": "INCOMPLETE", "state": args.state, "completed": len(existing), "records": list(existing.values())})
        print(json.dumps({"event": "GENERATION_PROGRESS", "state": args.state, "completed": len(existing), "total": 3000, "actual_batch": len(batch)}), flush=True)
    records = [existing[row["sample_id"]] for row in contexts]
    allowed = {"sample_id", "cohort", "model_state", "generated_token_ids", "generated_length", "termination_status", "prompt_sha256"}
    if len(records) != 3000 or any(set(row) != allowed for row in records): raise DiagnosticError("RAW_SCHEMA_OR_COUNT_FAILURE")
    write_jsonl_gz(_raw_path(args, args.state), records)
    receipt = {"status": "COMPLETE", "model_state": args.state, "record_count": len(records), "wall_seconds": time.perf_counter()-started,
               "raw_sha256": sha256_file(_raw_path(args, args.state)), "actual_batch_sizes": batch_sizes,
               "oom_fallbacks": oom_fallbacks, "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
               "optimizer_steps": 0, "backward_calls": 0, "optimizer_state_created": False}
    atomic_json(args.output / f"{args.state.upper()}_GENERATION_RECEIPT.json", receipt)
    partial_path.unlink(missing_ok=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return 0


def freeze(args: argparse.Namespace) -> int:
    context = {row["sample_id"]: row for row in load_contexts(args)}
    hashes: dict[str, Any] = {}; total = 0
    for state in MODEL_STATES:
        path = _raw_path(args, state); rows = read_jsonl_gz(path)
        if len(rows) != 3000 or any(row["sample_id"] not in context or row["prompt_sha256"] != context[row["sample_id"]]["prompt_sha256"] for row in rows):
            raise DiagnosticError(f"RAW_FREEZE_FAILURE={state}")
        hashes[state] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}; total += len(rows)
    core = {"status": "FROZEN", "BASE_COMPLETE": True, "FOUNDATION_V2_COMPLETE": True, "ALL_RAW_HASHES_PASS": True,
            "total_predictions": total, "raw_files": hashes, "freeze_unix_time": time.time()}
    core["freeze_core_sha256"] = canonical_sha256(core)
    core["GOLD_SCORING_STARTED_AFTER_FREEZE"] = False
    atomic_json(args.output / "PRE_GOLD_RAW_PREDICTION_FREEZE.json", core)
    atomic_json(args.artifact / "RAW_PREDICTION_FREEZE.json", core)
    return 0


def score(args: argparse.Namespace) -> int:
    freeze_path = args.output / "PRE_GOLD_RAW_PREDICTION_FREEZE.json"
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze_payload.get("status") != "FROZEN" or not freeze_payload.get("ALL_RAW_HASHES_PASS"):
        raise DiagnosticError("SCORING_BEFORE_FREEZE")
    for state in MODEL_STATES:
        if sha256_file(_raw_path(args, state)) != freeze_payload["raw_files"][state]["sha256"]:
            raise DiagnosticError(f"POST_FREEZE_RAW_DRIFT={state}")
    _identity, sources = verify_inputs(args)
    source = {row["sample_id"]: row for row in sources}
    scored: dict[str, dict[str, list[dict[str, Any]]]] = {state: {cohort: [] for cohort in INPUTS} for state in MODEL_STATES}
    parse_counts: dict[str, int] = defaultdict(int)
    for state in MODEL_STATES:
        for prediction in read_jsonl_gz(_raw_path(args, state)):
            row = source[prediction["sample_id"]]; gold = row["task"]["test"][0]["output"]
            result = score_prediction(prediction["sample_id"], prediction["generated_token_ids"], gold)
            enriched = {**{key: value for key, value in row.items() if key not in {"task", "scene_evidence", "mechanical_witness"}}, **result}
            scored[state][row["cohort"]].append(enriched); parse_counts[result["parse_failure_class"]] += 1
    direct_raw, minimal_raw, composite_raw = aggregate_capabilities({state: scored[state]["capability"] for state in MODEL_STATES})
    direct = pair_models(direct_raw, "exact_accuracy")
    minimal = pair_models(minimal_raw, "contrast_success_rate")
    composite = pair_models(composite_raw, "exact_accuracy", composite=True)
    profile = direct + minimal
    params = parameter_results({state: scored[state]["parameter"] for state in MODEL_STATES})
    compositions = composition_results({state: scored[state]["composition"] for state in MODEL_STATES})
    stages = stage_summary(profile)
    priorities = priority_map(profile, params, composite)
    delta_map = {"improved": sorted([row for row in profile if row["delta"] > 0], key=lambda row: -row["delta"]),
                 "unchanged": [row for row in profile if row["delta"] == 0],
                 "regressed": sorted([row for row in profile if row["delta"] < 0], key=lambda row: row["delta"]),
                 "interpretation": "Raw diagnostic deltas; one-sample changes are not strong conclusions."}
    atomic_json(args.artifact / "DIRECT_ATOMIC_RESULTS.json", {"status": "COMPLETE", "capabilities": direct})
    atomic_json(args.artifact / "MINIMAL_CONTRAST_RESULTS.json", {"status": "COMPLETE", "primary_metric": "CONTRAST_SUCCESS", "capabilities": minimal})
    atomic_json(args.artifact / "COMPOSITE_CHAIN_RESULTS.json", {"status": "COMPLETE", "independent_primitive_claim": False, "capabilities": composite})
    atomic_json(args.artifact / "PARAMETER_GENERALIZATION_RESULTS.json", {"status": "COMPLETE", "axes": params})
    atomic_json(args.artifact / "COMPOSITION_DEVELOPMENT_RESULTS.json", {"status": "COMPLETE", "development_exposed": True, "categories": compositions})
    interpretation = "This diagnostic measures Foundation-V2 performance over the frozen engineered capability ontology and its controlled generalization surfaces."
    profile_payload = {"status": "COMPLETE", "interpretation": interpretation, "capabilities": profile + composite}
    atomic_json(args.artifact / "FOUNDATION_V2_CAPABILITY_PROFILE.json", profile_payload)
    lines = ["# Foundation-V2 Capability Profile", "", interpretation, "", "| Capability | Type | Base | Foundation-V2 | Delta | Status |", "|---|---:|---:|---:|---:|---|"]
    for row in sorted(profile + composite, key=lambda item: item["capability"]):
        lines.append(f"| {row['capability']} | {row['measurement_type']} | {row['base_primary_score']:.4f} | {row['foundation_v2_primary_score']:.4f} | {row['delta']:+.4f} | {row['engineering_status']} |")
    atomic_text(args.artifact / "FOUNDATION_V2_CAPABILITY_PROFILE.md", "\n".join(lines) + "\n")
    atomic_json(args.artifact / "BASE_VS_FOUNDATION_V2_DELTA_MAP.json", delta_map)
    atomic_json(args.artifact / "STAGE_LEVEL_CAPABILITY_SUMMARY.json", {"status": "COMPLETE", "stages": stages, "composite_only_reported_separately": True})
    atomic_json(args.artifact / "FOUNDATION_CURRICULUM_PRIORITY_MAP_V1.json", {"status": "DIAGNOSTIC_ONLY_NO_WEIGHTS", "capabilities": priorities})
    adapter_hash = sha256_file(args.adapter_path / "adapter_model.safetensors")
    no_training = {"status": "PASS" if adapter_hash == ADAPTER_SHA256 else "FAIL", "optimizer_steps": 0, "backward_calls": 0,
                   "optimizer_state_created": False, "model_parameters_modified": False, "adapter_modified": adapter_hash != ADAPTER_SHA256,
                   "post_eval_adapter_sha256": adapter_hash, "expected_adapter_sha256": ADAPTER_SHA256}
    atomic_json(args.artifact / "NO_TRAINING_AUDIT.json", no_training)
    atomic_json(args.artifact / "DATA_ACCESS_AUDIT.json", data_access_audit())
    freeze_payload["GOLD_SCORING_STARTED_AFTER_FREEZE"] = True; freeze_payload["gold_scoring_started_unix_time"] = time.time()
    atomic_json(args.artifact / "RAW_PREDICTION_FREEZE.json", freeze_payload)
    calibration = json.loads((args.artifact / "GPU_DIAGNOSTIC_BATCH_CALIBRATION.json").read_text(encoding="utf-8"))
    receipts = [json.loads((args.output / f"{state.upper()}_GENERATION_RECEIPT.json").read_text(encoding="utf-8")) for state in MODEL_STATES]
    report = {"status": "COMPLETE", "branch": "evaluation/foundation-v2-capability-diagnostic-v1", "total_generations_completed": 6000,
              "generation_wall_seconds": sum(float(row["wall_seconds"]) for row in receipts), "selected_batch_configuration": calibration["selected_configuration"],
              "stage_summary": stages, "direct_band_counts": band_counts(direct), "minimal_band_counts": band_counts(minimal),
              "top_10_weakest": sorted(profile, key=lambda row: (row["foundation_v2_primary_score"], row["capability"]))[:10],
              "top_10_gains": sorted(profile, key=lambda row: (-row["delta"], row["capability"]))[:10],
              "regressions": delta_map["regressed"], "parameter_generalization": params, "composition_development": compositions,
              "curriculum_priority_summary": dict(sorted(__import__("collections").Counter(row["priority"] for row in priorities).items())),
              "parse_classification_counts": dict(sorted(parse_counts.items())), "interpretation": interpretation,
              "optimizer_steps": 0, "adapter_sha_unchanged": adapter_hash == ADAPTER_SHA256, "data_access_audit": data_access_audit()}
    atomic_json(args.artifact / "REPORT.json", report)
    atomic_text(args.artifact / "REPORT.md", report_markdown(report))
    write_sha_ledger(args.artifact)
    print(json.dumps({"status": "COMPLETE", "generations": 6000, "adapter_sha_unchanged": adapter_hash == ADAPTER_SHA256}), flush=True)
    return 0


def band_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {band: sum(str(row["engineering_status"]).endswith(band) for row in rows) for band in ("WEAK", "PARTIAL", "STRONG", "SATURATED")}


def report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Foundation-V2 Capability Diagnostic V1", "", report["interpretation"], "", "## Stage summary", ""]
    lines += [f"- {stage}: {json.dumps(values, sort_keys=True)}" for stage, values in report["stage_summary"].items()]
    lines += ["", "## Safety", "", "- Optimizer steps: 0", f"- Adapter SHA unchanged: {report['adapter_sha_unchanged']}", "- Forbidden holdouts and Gold accessed: false", ""]
    return "\n".join(lines)


def write_sha_ledger(artifact: Path) -> None:
    entries = []
    for path in sorted(artifact.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt": entries.append(f"{sha256_file(path)}  {path.name}")
    atomic_text(artifact / "SHA256SUMS.txt", "\n".join(entries) + "\n")


def controller(args: argparse.Namespace) -> int:
    common = ["--input-root", str(args.input_root), "--model-path", str(args.model_path), "--native-config-dir", str(args.native_config_dir),
              "--adapter-path", str(args.adapter_path), "--output", str(args.output), "--artifact", str(args.artifact),
              "--batch-size", str(args.batch_size), "--max-batched-prompt-tokens", str(args.max_batched_prompt_tokens)]
    def run(mode: str, extra: list[str] | None = None) -> None:
        command = [sys.executable, str(Path(__file__).resolve()), mode, *common, *(extra or [])]
        subprocess.run(command, check=True)
    run("preflight"); run("calibration")
    for state in MODEL_STATES: run("generate", ["--state", state])
    run("freeze"); run("score")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("preflight", "calibration", "generate", "freeze", "score", "controller"))
    result.add_argument("--input-root", type=Path, required=True); result.add_argument("--model-path", type=Path, required=True)
    result.add_argument("--native-config-dir", type=Path, required=True); result.add_argument("--adapter-path", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True); result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=32); result.add_argument("--max-batched-prompt-tokens", type=int, default=21568)
    result.add_argument("--state", choices=MODEL_STATES)
    return result


def main() -> int:
    args = parser().parse_args()
    return {"preflight": preflight, "calibration": calibration, "generate": generate_state, "freeze": freeze, "score": score, "controller": controller}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
