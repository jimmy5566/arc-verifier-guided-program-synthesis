"""Apply public-reference-style selection to an already frozen native pool.

No grid is generated here.  Model calls are teacher-forced likelihood scoring
of candidates which already existed in the input artifact.  The script never
imports challenge solutions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices

FROZEN = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"


def _original_scores(record: dict[str, Any]) -> dict[int, float]:
    indices = [int(value) for value in record.get("ranked_candidate_indices", ())]
    scores = [float(value) for value in record.get("candidate_scores", ())]
    count = len(record.get("candidates", ()))
    if len(indices) != count or len(scores) != count or set(indices) != set(range(count)):
        raise ValueError("candidate likelihood scores must cover every candidate exactly once")
    return dict(zip(indices, scores, strict=True))


def _cached_b_support_evidence(record: dict[str, Any], views: tuple[NativeAugmentation, ...]) -> list[PublicReferenceEvidence] | None:
    """Read task-local B evidence computed by the generation worker.

    A complete cache means the CPU portfolio phase needs no model process and
    no serial GPU tail.  Validation is deliberately strict so a legacy or
    partial artifact falls back to the historical scoring implementation.
    """
    candidates = list(record.get("candidates", ()))
    raw = record.get("b_support_evidence")
    if record.get("b_support_view_spec") != [view.to_dict() for view in views] or not isinstance(raw, list) or len(raw) != len(candidates):
        return None
    original = _original_scores(record)
    by_index = {int(item.get("candidate_index", -1)): item for item in raw if isinstance(item, dict)}
    if set(by_index) != set(range(len(candidates))):
        return None
    evidence: list[PublicReferenceEvidence] = []
    for index, candidate in enumerate(candidates):
        item = by_index[index]
        scores = item.get("view_negative_log_likelihoods")
        if not isinstance(scores, list) or len(scores) != len(views):
            return None
        if float(item.get("original_log_likelihood", float("nan"))) != original[index]:
            return None
        evidence.append(PublicReferenceEvidence(
            index=index,
            prediction_key=prediction_key(candidate["prediction"]),
            original_log_likelihood=original[index],
            view_negative_log_likelihoods=tuple(float(score) for score in scores),
            support_count=int(candidate.get("support_count", 1)),
        ))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("frozen", "challenge_path", "model_path", "native_config_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-window", type=int, default=16384)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--allow-deadline-partial", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen selection artifact")
    source = json.loads(args.frozen.read_text(encoding="utf-8"))
    records = source.get("records")
    accepted = {FROZEN, "DEADLINE_PARTIAL_CANDIDATES_FROZEN"} if args.allow_deadline_partial else {FROZEN}
    if source.get("status") not in accepted or not isinstance(records, dict) or not records:
        raise ValueError("requires a complete native candidate artifact frozen before exact scoring")
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        result = copy.deepcopy(source); result["records"] = {}; result["status"] = "PUBLIC_REFERENCE_SELECTION_PARTIAL_DEADLINE_FROZEN"; result["deadline_skipped_task_ids"] = sorted(records); result["public_reference_source_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest(); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return
    views = tuple(NativeAugmentation(geometry=geometry) for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    cached_by_task = {task_id: _cached_b_support_evidence(record, views) for task_id, record in records.items()}
    needs_model = any(value is None for value in cached_by_task.values())
    provider = None
    tasks = None
    if needs_model:
        # Legacy candidate artifacts do not include task-local B evidence.
        # Preserve their exact historical scoring path, still target-blind.
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider
        from inference.native_multiview_likelihood import candidate_view_scores
        tasks = load_dataset(args.challenge_path)
        provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device=args.device)
        provider.load()
    result = copy.deepcopy(source); result["records"] = {}; started = time.perf_counter(); skipped: list[str] = []
    for position, task_id in enumerate(sorted(records), 1):
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            skipped.extend(sorted(records)[position - 1:]); break
        record = copy.deepcopy(records[task_id]); candidates = list(record.get("candidates", ()))
        if not candidates or (tasks is not None and task_id not in tasks):
            raise ValueError(f"{task_id}: missing challenge or valid frozen candidates")
        evidence = cached_by_task[task_id]
        evidence_source = "task_local_inline_cache"
        if evidence is None:
            assert provider is not None and tasks is not None
            original = _original_scores(record); evidence = []
            for index, candidate in enumerate(candidates):
                scores = candidate_view_scores(provider, tasks[task_id], candidate["prediction"], views, context_window=args.context_window)
                evidence.append(PublicReferenceEvidence(index=index, prediction_key=prediction_key(candidate["prediction"]), original_log_likelihood=original[index], view_negative_log_likelihoods=tuple(-float(value) for value in scores), support_count=int(candidate.get("support_count", 1))))
            evidence_source = "legacy_serial_model_scoring"
        ranked = grouped_public_reference_ranking(evidence)
        attempts = two_attempt_indices(ranked, candidates)
        record["public_reference_selection"] = {
            "method": "equivalent_output_support_minus_mean_augmentation_view_nll",
            "views": [view.to_dict() for view in views],
            "legacy_support_limitation": all("support_count" not in candidate for candidate in candidates),
            "evidence": [{"candidate_index": item.index, "support_count": item.support_count, "mean_view_nll": item.mean_view_nll, "original_log_likelihood": item.original_log_likelihood} for item in evidence],
            "ranked_candidate_indices": ranked,
            "attempt_candidate_indices": attempts,
            "evidence_source": evidence_source,
        }
        result["records"][task_id] = record
        print(json.dumps({"event": "PUBLIC_REFERENCE_SELECTION_FROZEN", "task": f"{position}/{len(records)}", "task_id": task_id, "candidate_count": len(candidates), "attempt_count": len(attempts), "evidence_source": evidence_source}, sort_keys=True), flush=True)
    result["status"] = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" if not skipped and source.get("status") == FROZEN else "PUBLIC_REFERENCE_SELECTION_PARTIAL_DEADLINE_FROZEN"
    result["deadline_skipped_task_ids"] = skipped
    result["public_reference_source_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    result["public_reference_selection_runtime_seconds"] = time.perf_counter() - started
    result["public_reference_selection_protocol"] = "Existing native candidates only; teacher-forced scores in fixed reversible views; no candidate generation, target outputs, or task-specific rules."
    result["public_reference_evidence_mode"] = "task_local_inline_cache" if not needs_model else "mixed_or_legacy_serial_model_scoring"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
