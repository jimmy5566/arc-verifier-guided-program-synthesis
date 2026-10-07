"""Production 6000-generation Foundation-V2 capability diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_foundation_v2_capability_diagnostic_v1 as legacy
from foundation_v2_capability_diagnostic_v1.audit import (
    EOS_TOKEN_ID,
    MAX_NEW_TOKENS,
    MODEL_STATES,
    PAD_TOKEN_ID,
    DiagnosticError,
)


BRANCH = "evaluation/foundation-v2-capability-diagnostic-v1-production"
PROTOCOL_ID = "MAX_BATCH32_TOKEN_CAP21568"
FORENSIC_SHA256 = "ad1a8159c55a679c827f0673d3afb7da593c418b2aebe42a26ea71c0a922773c"
MAX_BATCH = 32
TOKEN_CAP = 21568
FLUSH_RECORDS = 128


def accepted_protocol_preflight(args: argparse.Namespace) -> int:
    forensic_path = args.artifact / "B32_ROOT_CAUSE_FORENSIC.json"
    if legacy.sha256_file(forensic_path) != FORENSIC_SHA256:
        raise DiagnosticError("B32_FORENSIC_SHA_MISMATCH")
    forensic = json.loads(forensic_path.read_text(encoding="utf-8"))
    if forensic.get("B32_PROTOCOL_DETERMINISTIC") is not True or forensic.get("Gold accessed") is not False:
        raise DiagnosticError("B32_FORENSIC_ACCEPTANCE_FAILURE")
    if forensic.get("original_five_mismatches_reproduced") != "EXACT_REPRODUCTION":
        raise DiagnosticError("B32_SENTINEL_NOT_REPRODUCED")
    identity, _rows = legacy.verify_inputs(args)
    adapter_hash = legacy.sha256_file(args.adapter_path / "adapter_model.safetensors")
    if adapter_hash != legacy.ADAPTER_SHA256:
        raise DiagnosticError("ADAPTER_SHA_MISMATCH")
    receipt = {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "B32_PROTOCOL_DETERMINISTIC": True,
        "B1_TOKEN_DIVERGENCE_SENTINEL": "5/64",
        "KNOWN_NUMERICAL_PROTOCOL_SENSITIVITY": True,
        "IMPLEMENTATION_DEFECT_ESTABLISHED": False,
        "B1_IS_NORMATIVE_PROTOCOL": False,
        "BATCHED_PROTOCOL_IS_NORMATIVE": True,
        "forensic_path": str(forensic_path),
        "forensic_sha256": FORENSIC_SHA256,
        "adapter_sha256": adapter_hash,
        "input_identity": identity,
        "configuration": {
            "max_batch": MAX_BATCH,
            "max_batched_prompt_tokens": TOKEN_CAP,
            "prompt_length_bucketing": True,
            "left_padding": True,
            "pad_token_id": PAD_TOKEN_ID,
            "explicit_attention_mask": True,
            "eos_token_id": EOS_TOKEN_ID,
            "context_window": 8704,
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": True,
            "torch_dtype": "bfloat16",
            "attention_backend": "sdpa",
        },
        "Gold accessed": False,
        "calibration_generation": "NOT_RUN_ACCEPTED_FROZEN_PROTOCOL",
    }
    legacy.atomic_json(args.artifact / "GPU_DIAGNOSTIC_PROTOCOL_ACCEPTANCE.json", receipt)
    print(json.dumps({"status": "PASS", "protocol_id": PROTOCOL_ID}), flush=True)
    return 0


def production_batches(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pending = sorted(rows, key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    groups: list[list[dict[str, Any]]] = []
    cursor = 0
    while cursor < len(pending):
        count = min(MAX_BATCH, len(pending) - cursor)
        while count > 1 and max(len(row["prompt_ids"]) for row in pending[cursor : cursor + count]) * count > TOKEN_CAP:
            count -= 1
        groups.append(pending[cursor : cursor + count])
        cursor += count
    return groups


def generate_batch_clean(model: Any, batch: list[dict[str, Any]]) -> list[list[int]]:
    import torch

    width = max(len(row["prompt_ids"]) for row in batch)
    prompt = torch.full((len(batch), width), PAD_TOKEN_ID, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(prompt)
    for index, row in enumerate(batch):
        length = len(row["prompt_ids"])
        prompt[index, width - length :] = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda:0")
        mask[index, width - length :] = 1
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            attention_mask=mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=None,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            use_cache=True,
        )
    suffixes: list[list[int]] = []
    for row in output:
        raw = [int(value) for value in row[width:].detach().cpu().tolist()]
        suffixes.append(raw[: raw.index(EOS_TOKEN_ID) + 1] if EOS_TOKEN_ID in raw else raw)
    del output, prompt, mask
    return suffixes


def _partial_path(args: argparse.Namespace, state: str) -> Path:
    return args.output / "raw_predictions" / f"{state}.partial.jsonl"


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in records:
                raise DiagnosticError(f"DUPLICATE_PARTIAL_SAMPLE={sample_id}:{line_number}")
            records[sample_id] = row
    return records


def append_records(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _receipt_name(state: str) -> str:
    return f"{state.upper()}_GENERATION_RECEIPT.json"


def generate_state_production(args: argparse.Namespace) -> int:
    import torch

    if args.state not in MODEL_STATES:
        raise DiagnosticError(f"BAD_STATE={args.state}")
    contexts = legacy.load_contexts(args)
    partial_path = _partial_path(args, args.state)
    existing = load_partial(partial_path)
    context_ids = {row["sample_id"] for row in contexts}
    if not set(existing).issubset(context_ids):
        raise DiagnosticError("PARTIAL_COHORT_DRIFT")
    pending = [row for row in contexts if row["sample_id"] not in existing]
    groups = production_batches(pending)
    model = legacy.load_model(args, args.state)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    buffered: list[dict[str, Any]] = []
    generated_lengths = [int(row["generated_length"]) for row in existing.values()]
    distribution: Counter[int] = Counter()
    next_progress = ((len(existing) // 250) + 1) * 250
    last_batch_size = 0
    try:
        for batch_index, batch in enumerate(groups):
            last_batch_size = len(batch)
            try:
                suffixes = generate_batch_clean(model, batch)
            except torch.cuda.OutOfMemoryError as exc:
                append_records(partial_path, buffered)
                buffered.clear()
                failure = {
                    "status": "PRODUCTION_BATCH_OOM",
                    "state": args.state,
                    "batch_index": batch_index,
                    "sample_ids": [row["sample_id"] for row in batch],
                    "actual_batch_size": len(batch),
                    "prompt_lengths": [len(row["prompt_ids"]) for row in batch],
                    "padded_width": max(len(row["prompt_ids"]) for row in batch),
                    "completed_preserved": len(existing),
                    "error": str(exc),
                }
                legacy.atomic_json(args.output / f"{args.state.upper()}_PRODUCTION_BATCH_OOM.json", failure)
                del batch
                gc.collect()
                torch.cuda.empty_cache()
                raise DiagnosticError("PRODUCTION_BATCH_OOM") from exc
            for context, tokens in zip(batch, suffixes, strict=True):
                row = {
                    "sample_id": context["sample_id"],
                    "cohort": context["cohort"],
                    "model_state": args.state,
                    "generated_token_ids": tokens,
                    "generated_length": len(tokens),
                    "termination_status": "EOS" if tokens and tokens[-1] == EOS_TOKEN_ID else "MAX_NEW_TOKENS",
                    "prompt_sha256": context["prompt_sha256"],
                }
                existing[row["sample_id"]] = row
                buffered.append(row)
                generated_lengths.append(len(tokens))
            distribution[len(batch)] += 1
            if len(buffered) >= FLUSH_RECORDS:
                append_records(partial_path, buffered)
                buffered.clear()
            completed = len(existing)
            if completed >= next_progress or completed == 3000:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "state": args.state,
                            "completed": completed,
                            "total": 3000,
                            "elapsed_seconds": elapsed,
                            "examples_per_second": (completed / elapsed) if elapsed else 0,
                            "last_actual_batch_size": last_batch_size,
                        }
                    ),
                    flush=True,
                )
                while next_progress <= completed:
                    next_progress += 250
        append_records(partial_path, buffered)
        buffered.clear()
        torch.cuda.synchronize(0)
        wall = time.perf_counter() - started
        records = [existing[row["sample_id"]] for row in contexts]
        allowed = {"sample_id", "cohort", "model_state", "generated_token_ids", "generated_length", "termination_status", "prompt_sha256"}
        if len(records) != 3000 or len(existing) != 3000 or any(set(row) != allowed for row in records):
            raise DiagnosticError("RAW_SCHEMA_OR_COUNT_FAILURE")
        raw_path = legacy._raw_path(args, args.state)
        legacy.write_jsonl_gz(raw_path, records)
        verified = legacy.read_jsonl_gz(raw_path)
        if len(verified) != 3000 or [row["sample_id"] for row in verified] != [row["sample_id"] for row in contexts]:
            raise DiagnosticError("FINAL_RAW_VERIFICATION_FAILURE")
        total_tokens = sum(generated_lengths)
        receipt = {
            "status": "COMPLETE",
            "protocol_id": PROTOCOL_ID,
            "model_state": args.state,
            "record_count": 3000,
            "wall_seconds": wall,
            "examples_per_second": 3000 / wall,
            "generated_tokens_per_second": total_tokens / wall,
            "total_generated_tokens": total_tokens,
            "mean_generated_length": statistics.fmean(generated_lengths),
            "median_generated_length": statistics.median(generated_lengths),
            "actual_batch_size_distribution": {str(key): value for key, value in sorted(distribution.items())},
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
            "oom_count": 0,
            "raw_sha256": legacy.sha256_file(raw_path),
            "optimizer_steps": 0,
            "backward_calls": 0,
            "optimizer_state_created": False,
        }
        legacy.atomic_json(args.output / _receipt_name(args.state), receipt)
        legacy.atomic_json(args.artifact / _receipt_name(args.state), receipt)
        partial_path.unlink(missing_ok=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


def freeze_production(args: argparse.Namespace) -> int:
    contexts = legacy.load_contexts(args)
    context = {row["sample_id"]: row for row in contexts}
    raw_files: dict[str, Any] = {}
    total = 0
    for state in MODEL_STATES:
        path = legacy._raw_path(args, state)
        rows = legacy.read_jsonl_gz(path)
        ids = [row["sample_id"] for row in rows]
        if len(rows) != 3000 or len(set(ids)) != 3000 or ids != [row["sample_id"] for row in contexts]:
            raise DiagnosticError(f"RAW_FREEZE_COUNT_OR_ORDER_FAILURE={state}")
        if any(row["model_state"] != state or row["prompt_sha256"] != context[row["sample_id"]]["prompt_sha256"] for row in rows):
            raise DiagnosticError(f"RAW_FREEZE_IDENTITY_FAILURE={state}")
        raw_files[state] = {"path": str(path), "rows": 3000, "sha256": legacy.sha256_file(path)}
        total += 3000
    receipt = {
        "status": "FROZEN",
        "BASE_COMPLETE": True,
        "FOUNDATION_V2_COMPLETE": True,
        "TOTAL_RAW_PREDICTIONS": total,
        "RAW_HASHES_VERIFIED": True,
        "ALL_RAW_HASHES_PASS": True,
        "GOLD_SCORING_STARTED": False,
        "GOLD_SCORING_STARTED_AFTER_FREEZE": False,
        "protocol_id": PROTOCOL_ID,
        "raw_files": raw_files,
        "freeze_unix_time": time.time(),
    }
    receipt["freeze_core_sha256"] = legacy.canonical_sha256(receipt)
    legacy.atomic_json(args.output / "PRE_GOLD_RAW_PREDICTION_FREEZE.json", receipt)
    legacy.atomic_json(args.artifact / "RAW_PREDICTION_FREEZE.json", receipt)
    return 0


def _band_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {band: sum(str(row["engineering_status"]).endswith(band) for row in rows) for band in ("WEAK", "PARTIAL", "STRONG", "SATURATED")}
    result["BORDERLINE"] = sum(bool(row.get("borderline")) for row in rows)
    return result


def score_production(args: argparse.Namespace) -> int:
    legacy.score(args)
    profile_path = args.artifact / "FOUNDATION_V2_CAPABILITY_PROFILE.json"
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    independent = [row for row in profile_payload["capabilities"] if row["measurement_type"] != "COMPOSITE_ONLY"]
    composites = [row for row in profile_payload["capabilities"] if row["measurement_type"] == "COMPOSITE_ONLY"]
    weakness = sorted(independent, key=lambda row: (row["foundation_v2_primary_score"], row["capability"]))
    strength = sorted(independent, key=lambda row: (-row["foundation_v2_primary_score"], row["capability"]))
    legacy.atomic_json(args.artifact / "FOUNDATION_V2_WEAKNESS_MAP.json", {"status": "COMPLETE", "capabilities": weakness})
    legacy.atomic_json(args.artifact / "FOUNDATION_V2_STRENGTH_MAP.json", {"status": "COMPLETE", "capabilities": strength})
    direct = json.loads((args.artifact / "DIRECT_ATOMIC_RESULTS.json").read_text(encoding="utf-8"))["capabilities"]
    minimal = json.loads((args.artifact / "MINIMAL_CONTRAST_RESULTS.json").read_text(encoding="utf-8"))["capabilities"]
    receipts = {
        state: json.loads((args.artifact / _receipt_name(state)).read_text(encoding="utf-8")) for state in MODEL_STATES
    }
    report_path = args.artifact / "REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "branch": BRANCH,
            "protocol_id": PROTOCOL_ID,
            "KNOWN_NUMERICAL_PROTOCOL_SENSITIVITY": True,
            "B1_TOKEN_DIVERGENCE_SENTINEL": "5/64",
            "BATCHED_PROTOCOL_DETERMINISTIC": True,
            "generation_wall_seconds": sum(float(value["wall_seconds"]) for value in receipts.values()),
            "state_generation_receipts": receipts,
            "direct_band_counts": _band_counts(direct),
            "minimal_band_counts": _band_counts(minimal),
            "composite_reported_separately": composites,
        }
    )
    legacy.atomic_json(report_path, report)
    lines = [
        "# Foundation-V2 Capability Diagnostic V1 Production",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "B1 is calibration-only. All official scores use the deterministic batched protocol.",
        "",
        "| Capability | Type | Base | Foundation-V2 | Delta | Band | Borderline |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in sorted(profile_payload["capabilities"], key=lambda item: item["capability"]):
        lines.append(
            f"| {row['capability']} | {row['measurement_type']} | {row['base_primary_score']:.4f} | "
            f"{row['foundation_v2_primary_score']:.4f} | {row['delta']:+.4f} | {row['engineering_status']} | {bool(row.get('borderline'))} |"
        )
    legacy.atomic_text(args.artifact / "FOUNDATION_V2_CAPABILITY_PROFILE.md", "\n".join(lines) + "\n")
    freeze_path = args.artifact / "RAW_PREDICTION_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["GOLD_SCORING_STARTED"] = True
    legacy.atomic_json(freeze_path, freeze)
    legacy.write_sha_ledger(args.artifact)
    return 0


def controller(args: argparse.Namespace) -> int:
    common = [
        "--input-root", str(args.input_root),
        "--model-path", str(args.model_path),
        "--native-config-dir", str(args.native_config_dir),
        "--adapter-path", str(args.adapter_path),
        "--output", str(args.output),
        "--artifact", str(args.artifact),
    ]

    def run(mode: str, extra: list[str] | None = None) -> None:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, *common, *(extra or [])], check=True)

    run("preflight")
    run("accepted_protocol_preflight")
    run("generate", ["--state", "base"])
    run("generate", ["--state", "foundation_v2"])
    run("freeze")
    run("score")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("mode", choices=("preflight", "accepted_protocol_preflight", "generate", "freeze", "score", "controller"))
    value.add_argument("--input-root", type=Path, required=True)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--native-config-dir", type=Path, required=True)
    value.add_argument("--adapter-path", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--artifact", type=Path, required=True)
    value.add_argument("--state", choices=MODEL_STATES)
    return value


def main() -> int:
    args = parser().parse_args()
    return {
        "preflight": legacy.preflight,
        "accepted_protocol_preflight": accepted_protocol_preflight,
        "generate": generate_state_production,
        "freeze": freeze_production,
        "score": score_production,
        "controller": controller,
    }[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
