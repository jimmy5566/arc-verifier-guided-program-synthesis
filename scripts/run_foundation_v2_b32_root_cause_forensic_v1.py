"""Explain B1-versus-normal-B32 token differences without testing lower batches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_foundation_v2_capability_diagnostic_v1 import (
    ADAPTER_SHA256,
    DiagnosticError,
    atomic_json,
    calibration_selection,
    load_contexts,
    load_model,
    sha256_file,
)
from run_foundation_v2_batch_recovery_v1 import batches


NORMAL_MAX_BATCH = 32
NEAR_TIE_MARGIN = 0.05
EXPECTED_MISMATCHES = (
    "DIAGNOSTIC_V1_2:ff68d59469e5ba1fea3ce1d0",
    "PARAM_SURFACE_V1_2:object_count:diagnostic:018",
    "PARAM_SURFACE_V1_2:object_position:diagnostic:031",
    "COMPOSITION_DEV_V1_2:capv3:4b433153812ecc3d40:001",
    "COMPOSITION_DEV_V1_2:capv3:dbdcca62cc0200da4e:004",
)


def _tokens_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    if left == right:
        return None
    shared = min(len(left), len(right))
    return next((index for index in range(shared) if left[index] != right[index]), shared)


class Capture:
    def __init__(self, rows: dict[int, str]) -> None:
        self.rows = rows
        self.values: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        import torch

        for index, sample_id in self.rows.items():
            values, ids = torch.topk(scores[index].detach().float().cpu(), k=5)
            logits = [float(value) for value in values.tolist()]
            self.values[sample_id].append(
                {
                    "top5_token_ids": [int(value) for value in ids.tolist()],
                    "top5_logits": logits,
                    "top1_token_id": int(ids[0]),
                    "top2_token_id": int(ids[1]),
                    "top1_top2_margin": float(logits[0] - logits[1]),
                }
            )
        return scores


def generate(
    model: Any,
    batch: list[dict[str, Any]],
    *,
    capture_ids: set[str] | None = None,
    forced_width: int | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import LogitsProcessorList

    natural_width = max(len(row["prompt_ids"]) for row in batch)
    width = natural_width if forced_width is None else int(forced_width)
    if width < natural_width:
        raise DiagnosticError(f"FORCED_WIDTH_TOO_SMALL={width}<{natural_width}")
    prompt = torch.full((len(batch), width), 13, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(prompt)
    for index, row in enumerate(batch):
        length = len(row["prompt_ids"])
        prompt[index, width - length :] = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda:0")
        mask[index, width - length :] = 1
    capture = Capture(
        {
            index: row["sample_id"]
            for index, row in enumerate(batch)
            if capture_ids and row["sample_id"] in capture_ids
        }
    )
    processors = LogitsProcessorList([capture]) if capture.rows else None
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt,
            attention_mask=mask,
            max_new_tokens=932,
            do_sample=False,
            num_beams=1,
            eos_token_id=15,
            pad_token_id=13,
            use_cache=True,
            logits_processor=processors,
        )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    outputs: dict[str, list[int]] = {}
    for index, row in enumerate(output):
        raw = [int(value) for value in row[width:].detach().cpu().tolist()]
        outputs[batch[index]["sample_id"]] = raw[: raw.index(15) + 1] if 15 in raw else raw
    result = {
        "status": "PASS",
        "outputs": outputs,
        "captures": dict(capture.values),
        "wall_seconds": elapsed,
        "padded_width": width,
        "natural_width": natural_width,
        "actual_batch_size": len(batch),
        "prompt_lengths": [len(row["prompt_ids"]) for row in batch],
        "total_padded_prompt_tokens": width * len(batch),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    del output, prompt, mask
    return result


def safe_generate(model: Any, batch: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    import torch

    try:
        return generate(model, batch, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        message = str(exc)
        return {
            "status": "OOM",
            "error": message,
            "actual_batch_size": len(batch),
            "prompt_lengths": [len(row["prompt_ids"]) for row in batch],
            "padded_width": kwargs.get("forced_width") or max(len(row["prompt_ids"]) for row in batch),
        }
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def normal_b32_protocol(model: Any, rows: list[dict[str, Any]], token_cap: int) -> dict[str, Any]:
    import torch

    outputs: dict[str, list[int]] = {}
    captures: dict[str, list[dict[str, Any]]] = {}
    batch_records: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    started = time.perf_counter()
    groups = batches(rows, NORMAL_MAX_BATCH, token_cap)
    for index, group in enumerate(groups):
        run = safe_generate(model, group, capture_ids={row["sample_id"] for row in group})
        record = {
            "batch_index": index,
            "sample_ids": [row["sample_id"] for row in group],
            "actual_batch_size": len(group),
            "prompt_lengths": [len(row["prompt_ids"]) for row in group],
            "padded_width": run["padded_width"],
            "total_padded_prompt_tokens": run.get("total_padded_prompt_tokens", run["padded_width"] * len(group)),
            "peak_allocated_vram_bytes": run.get("peak_allocated_vram_bytes"),
            "peak_reserved_vram_bytes": run.get("peak_reserved_vram_bytes"),
            "status": run["status"],
        }
        batch_records.append(record)
        print(json.dumps({"event": "NORMAL_B32_BATCH", **record}), flush=True)
        if run["status"] != "PASS":
            return {"status": "FAIL_OOM", "batch_records": batch_records, "wall_seconds": time.perf_counter() - started}
        outputs.update(run["outputs"])
        captures.update(run["captures"])
    return {
        "status": "PASS",
        "outputs": outputs,
        "captures": captures,
        "batch_records": batch_records,
        "wall_seconds": time.perf_counter() - started,
    }


def _normalize_capture(capture: list[dict[str, Any]], generated_tokens: list[int]) -> list[dict[str, Any]]:
    result = []
    for entry, selected_token_id in zip(capture, generated_tokens):
        normalized = dict(entry)
        ids = normalized["top5_token_ids"]
        logits = normalized["top5_logits"]
        if selected_token_id in ids:
            selected_index = ids.index(selected_token_id)
            alternatives = [(logit, token) for token, logit in zip(ids, logits) if token != selected_token_id]
            top2_logit, top2_token_id = max(alternatives, key=lambda item: item[0])
            normalized.update(
                {
                    "topk_rank1_token_id": normalized["top1_token_id"],
                    "top1_token_id": selected_token_id,
                    "top2_token_id": top2_token_id,
                    "top1_top2_margin": float(logits[selected_index] - top2_logit),
                    "top1_token_source": "GREEDY_GENERATED_TOKEN",
                }
            )
        result.append(normalized)
    return result


def _condition_summary(run: dict[str, Any], sample_id: str, reference: list[int], capture_id: str | None = None) -> dict[str, Any]:
    summary = {key: value for key, value in run.items() if key not in {"outputs", "captures"}}
    if run["status"] != "PASS":
        return summary
    key = capture_id or sample_id
    candidate = run["outputs"][key]
    capture = _normalize_capture(run["captures"].get(key, []), candidate)
    summary.update(
        {
            "token_sha256": _tokens_hash(candidate),
            "matches_b1": candidate == reference,
            "first_divergence": _first_divergence(reference, candidate),
            "generated_token_ids": candidate,
            "capture": capture,
        }
    )
    return summary


def normalize_existing_report(report: dict[str, Any]) -> dict[str, Any]:
    """Normalize tied top-k display fields from already-frozen generated tokens."""
    for sample in report["samples"]:
        summaries = [sample["B1_NATIVE"]]
        for name, value in sample["conditions"].items():
            summaries.extend(value if name == "BN_ORIGINAL_PEERS_REPEAT" else [value])
        for summary in summaries:
            if summary.get("status") == "PASS":
                summary["capture"] = _normalize_capture(summary.get("capture", []), summary["generated_token_ids"])
        sample["first_divergence_logit_comparisons"] = {
            name: _logit_comparison(sample["B1_NATIVE"], value)
            for name, value in sample["conditions"].items()
            if name != "BN_ORIGINAL_PEERS_REPEAT"
        }
        sample["root_cause_classification"] = _labels(sample["conditions"], sample["B1_NATIVE"])
    report["top1_token_semantics"] = "GREEDY_GENERATED_TOKEN; topk_rank1_token_id preserves torch.topk tie ordering"
    return report


def _logit_comparison(b1: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any] | None:
    position = condition.get("first_divergence")
    if position is None:
        return None
    if position >= len(b1["capture"]) or position >= len(condition.get("capture", [])):
        return {"position": position, "status": "CAPTURE_OUT_OF_RANGE"}
    left = b1["capture"][position]
    right = condition["capture"][position]
    left_map = dict(zip(left["top5_token_ids"], left["top5_logits"]))
    right_map = dict(zip(right["top5_token_ids"], right["top5_logits"]))
    token_union = sorted(set(left_map) | set(right_map))
    return {
        "position": position,
        "shared_prefix_sha256": _tokens_hash(b1["generated_token_ids"][:position]),
        "b1": left,
        "condition": right,
        "absolute_logit_differences": {
            str(token): abs(left_map[token] - right_map[token])
            for token in token_union
            if token in left_map and token in right_map
        },
    }


def _labels(conditions: dict[str, Any], b1: dict[str, Any]) -> list[str]:
    padded = conditions["B1_PADDED_TO_W"]
    self_native = conditions["BN_SELF_NATIVE"]
    self_padded = conditions["BN_SELF_PADDED_TO_W"]
    peers = conditions["BN_ORIGINAL_PEERS"]
    reversed_peers = conditions["BN_ORIGINAL_PEERS_REVERSED"]
    labels: list[str] = []
    if padded.get("status") == "PASS" and not padded.get("matches_b1"):
        labels.append("PADDING_WIDTH_EFFECT")
    if padded.get("matches_b1") and self_native.get("status") == "PASS" and not self_native.get("matches_b1"):
        labels.append("BATCH_DIMENSION_NUMERICAL_EFFECT")
    if self_native.get("matches_b1") and self_padded.get("matches_b1") and peers.get("status") == "PASS" and not peers.get("matches_b1"):
        labels.append("HETEROGENEOUS_BATCH_NUMERICAL_EFFECT")
    if peers.get("status") == reversed_peers.get("status") == "PASS" and peers.get("token_sha256") != reversed_peers.get("token_sha256"):
        labels.append("ORDER_SENSITIVE_BATCH_EFFECT")
    logit = _logit_comparison(b1, peers)
    if logit and logit.get("status") != "CAPTURE_OUT_OF_RANGE":
        if min(logit["b1"]["top1_top2_margin"], logit["condition"]["top1_top2_margin"]) <= NEAR_TIE_MARGIN:
            labels.append("NUMERICAL_NEAR_TIE")
    return labels or ["UNRESOLVED"]


def forensic_one(
    model: Any,
    sample_id: str,
    contexts: dict[str, dict[str, Any]],
    reference: dict[str, list[int]],
    normal: dict[str, Any],
) -> dict[str, Any]:
    row = contexts[sample_id]
    batch_record = next(record for record in normal["batch_records"] if sample_id in record["sample_ids"])
    peers = [contexts[value] for value in batch_record["sample_ids"]]
    n, width = batch_record["actual_batch_size"], batch_record["padded_width"]
    b1_run = safe_generate(model, [row], capture_ids={sample_id})
    if b1_run["status"] != "PASS" or b1_run["outputs"][sample_id] != reference[sample_id]:
        raise DiagnosticError(f"B1_LOGIT_TRACE_DRIFT={sample_id}")
    b1 = _condition_summary(b1_run, sample_id, reference[sample_id])

    padded = safe_generate(model, [row], capture_ids={sample_id}, forced_width=width)
    self_native_rows = [{**row, "sample_id": sample_id}] + [
        {**row, "sample_id": f"{sample_id}#self{index}"} for index in range(1, n)
    ]
    self_native = safe_generate(model, self_native_rows, capture_ids={sample_id})
    self_padded = safe_generate(model, self_native_rows, capture_ids={sample_id}, forced_width=width)
    original = safe_generate(model, peers, capture_ids={sample_id})
    repeats = [safe_generate(model, peers, capture_ids={sample_id}) for _ in range(3)]
    reversed_original = safe_generate(model, list(reversed(peers)), capture_ids={sample_id})
    conditions = {
        "B1_PADDED_TO_W": _condition_summary(padded, sample_id, reference[sample_id]),
        "BN_SELF_NATIVE": _condition_summary(self_native, sample_id, reference[sample_id]),
        "BN_SELF_PADDED_TO_W": _condition_summary(self_padded, sample_id, reference[sample_id]),
        "BN_ORIGINAL_PEERS": _condition_summary(original, sample_id, reference[sample_id]),
        "BN_ORIGINAL_PEERS_REPEAT": [
            _condition_summary(run, sample_id, reference[sample_id]) for run in repeats
        ],
        "BN_ORIGINAL_PEERS_REVERSED": _condition_summary(reversed_original, sample_id, reference[sample_id]),
    }
    repeat_hashes = [value.get("token_sha256") for value in conditions["BN_ORIGINAL_PEERS_REPEAT"]]
    token_repeatability = all(value.get("status") == "PASS" for value in conditions["BN_ORIGINAL_PEERS_REPEAT"]) and len(set(repeat_hashes)) == 1
    logit_comparisons = {
        name: _logit_comparison(b1, value)
        for name, value in conditions.items()
        if name != "BN_ORIGINAL_PEERS_REPEAT"
    }
    return {
        "sample_id": sample_id,
        "original_normal_b32": {
            "N": n,
            "W": width,
            "batch_index": batch_record["batch_index"],
            "peer_sample_ids": batch_record["sample_ids"],
            "prompt_lengths": batch_record["prompt_lengths"],
            "total_padded_prompt_tokens": batch_record["total_padded_prompt_tokens"],
        },
        "B1_NATIVE": b1,
        "conditions": conditions,
        "first_divergence_logit_comparisons": logit_comparisons,
        "TOKEN_REPEATABILITY": "PASS" if token_repeatability else "FAIL",
        "root_cause_classification": _labels(conditions, b1),
    }


def main() -> int:
    args = parser().parse_args()
    contexts_list = load_contexts(args)
    selected = calibration_selection(contexts_list)
    ids = [row["sample_id"] for row in selected]
    contexts = {row["sample_id"]: row for row in contexts_list}
    reference_path = args.output / "B1_REFERENCE_RECOVERY.json"
    frozen = json.loads(reference_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN" or frozen.get("record_count") != 64:
        raise DiagnosticError("B1_REFERENCE_NOT_FROZEN_64")
    reference = {row["sample_id"]: [int(value) for value in row["generated_token_ids"]] for row in frozen["records"]}
    if set(reference) != set(ids):
        raise DiagnosticError("B1_REFERENCE_COHORT_DRIFT")
    model = load_model(args, "foundation_v2")
    normal = normal_b32_protocol(model, selected, args.max_batched_prompt_tokens)
    if normal["status"] != "PASS" or set(normal.get("outputs", {})) != set(ids):
        raise DiagnosticError("NORMAL_B32_PROTOCOL_FAILED")
    mismatches = [sample_id for sample_id in ids if normal["outputs"][sample_id] != reference[sample_id]]
    mismatch_set_status = "EXACT_REPRODUCTION" if set(mismatches) == set(EXPECTED_MISMATCHES) else "DIFFERENT"
    print(json.dumps({"event": "NORMAL_B32_COMPLETE", "mismatches": mismatches, "status": mismatch_set_status}), flush=True)
    results = []
    for index, sample_id in enumerate(mismatches, 1):
        results.append(forensic_one(model, sample_id, contexts, reference, normal))
        print(json.dumps({"event": "FORENSIC_SAMPLE", "completed": index, "total": len(mismatches), "sample_id": sample_id}), flush=True)
    deterministic = bool(results) and all(row["TOKEN_REPEATABILITY"] == "PASS" for row in results)
    report = {
        "status": "COMPLETE",
        "experiment": "FOUNDATION_V2_B32_ROOT_CAUSE_FORENSIC_V1",
        "normal_protocol": {
            "max_batch": NORMAL_MAX_BATCH,
            "max_batched_prompt_tokens": args.max_batched_prompt_tokens,
            "prompt_length_bucketing": True,
            "left_padding": True,
            "explicit_attention_mask": True,
            "batch_records": normal["batch_records"],
            "wall_seconds": normal["wall_seconds"],
        },
        "expected_mismatch_ids": list(EXPECTED_MISMATCHES),
        "observed_mismatch_ids": mismatches,
        "original_five_mismatches_reproduced": mismatch_set_status,
        "samples": results,
        "B32_PROTOCOL_DETERMINISTIC": deterministic,
        "near_tie_margin_threshold": NEAR_TIE_MARGIN,
        "lower_batch_ladder": "NOT_RUN_BY_PRIORITY_CHANGE",
        "Gold accessed": False,
        "full 6000 generation started": False,
        "b1_reference": {"path": str(reference_path), "sha256": sha256_file(reference_path), "records": 64},
        "adapter_sha256": sha256_file(args.adapter_path / "adapter_model.safetensors"),
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    if report["adapter_sha256"] != ADAPTER_SHA256:
        raise DiagnosticError("ADAPTER_DRIFT")
    atomic_json(args.artifact / "B32_ROOT_CAUSE_FORENSIC.json", report)
    print(json.dumps({"status": "COMPLETE", "mismatches": len(mismatches), "B32_PROTOCOL_DETERMINISTIC": deterministic}), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--adapter-path", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--artifact", type=Path, required=True)
    value.add_argument("--max-batched-prompt-tokens", type=int, default=21568)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
