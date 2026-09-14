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

from arc.io import load_dataset
from inference.nvarc_native import NVARCNativeProvider
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices
from inference.native_multiview_likelihood import candidate_view_scores

FROZEN = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"


def _original_scores(record: dict[str, Any]) -> dict[int, float]:
    indices = [int(value) for value in record.get("ranked_candidate_indices", ())]
    scores = [float(value) for value in record.get("candidate_scores", ())]
    count = len(record.get("candidates", ()))
    if len(indices) != count or len(scores) != count or set(indices) != set(range(count)):
        raise ValueError("candidate likelihood scores must cover every candidate exactly once")
    return dict(zip(indices, scores, strict=True))


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
    # Crucially, task challenges carry train pairs/test inputs only; no target
    # solutions path exists in this CLI.
    tasks = load_dataset(args.challenge_path)
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        result = copy.deepcopy(source); result["records"] = {}; result["status"] = "PUBLIC_REFERENCE_SELECTION_PARTIAL_DEADLINE_FROZEN"; result["deadline_skipped_task_ids"] = sorted(records); result["public_reference_source_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest(); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device=args.device)
    provider.load()
    views = tuple(NativeAugmentation(geometry=geometry) for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    result = copy.deepcopy(source); result["records"] = {}; started = time.perf_counter(); skipped: list[str] = []
    for position, task_id in enumerate(sorted(records), 1):
        if args.deadline_unix is not None and time.time() >= args.deadline_unix:
            skipped.extend(sorted(records)[position - 1:]); break
        record = copy.deepcopy(records[task_id]); candidates = list(record.get("candidates", ()))
        if task_id not in tasks or not candidates:
            raise ValueError(f"{task_id}: missing challenge or valid frozen candidates")
        original = _original_scores(record)
        evidence: list[PublicReferenceEvidence] = []
        for index, candidate in enumerate(candidates):
            scores = candidate_view_scores(provider, tasks[task_id], candidate["prediction"], views, context_window=args.context_window)
            evidence.append(PublicReferenceEvidence(
                index=index,
                prediction_key=prediction_key(candidate["prediction"]),
                original_log_likelihood=original[index],
                view_negative_log_likelihoods=tuple(-float(value) for value in scores),
                # Legacy baseline discarded pre-dedup repetitions.  Retaining
                # one here is honest; C preserves actual group support.
                support_count=int(candidate.get("support_count", 1)),
            ))
        ranked = grouped_public_reference_ranking(evidence)
        attempts = two_attempt_indices(ranked, candidates)
        record["public_reference_selection"] = {
            "method": "equivalent_output_support_minus_mean_augmentation_view_nll",
            "views": [view.to_dict() for view in views],
            "legacy_support_limitation": all("support_count" not in candidate for candidate in candidates),
            "evidence": [{"candidate_index": item.index, "support_count": item.support_count, "mean_view_nll": item.mean_view_nll, "original_log_likelihood": item.original_log_likelihood} for item in evidence],
            "ranked_candidate_indices": ranked,
            "attempt_candidate_indices": attempts,
        }
        result["records"][task_id] = record
        print(json.dumps({"event": "PUBLIC_REFERENCE_SELECTION_FROZEN", "task": f"{position}/{len(records)}", "task_id": task_id, "candidate_count": len(candidates), "attempt_count": len(attempts)}, sort_keys=True), flush=True)
    result["status"] = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" if not skipped and source.get("status") == FROZEN else "PUBLIC_REFERENCE_SELECTION_PARTIAL_DEADLINE_FROZEN"
    result["deadline_skipped_task_ids"] = skipped
    result["public_reference_source_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    result["public_reference_selection_runtime_seconds"] = time.perf_counter() - started
    result["public_reference_selection_protocol"] = "Existing native candidates only; teacher-forced scores in fixed reversible views; no candidate generation, target outputs, or task-specific rules."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
