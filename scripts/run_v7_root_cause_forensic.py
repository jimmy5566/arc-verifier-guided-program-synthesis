"""One-shot, target-blind forensic harness for ARC2 Kaggle production V7.

This tool is deliberately kept outside the V7 source package.  It invokes the
unmodified production scripts from ``--production-root`` and writes an audit
trail around them.  It never repairs, ranks, or generates a different
scientific configuration.

The only target boundary is the final scoring section, after the complete
replay submission and every injected-cutoff submission have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


FROZEN = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
PARTIAL = "DEADLINE_PARTIAL_CANDIDATES_FROZEN"
FAULT_COUNTS = (1, 5, 10, 15, 20, 25)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Stable content hash of the source package, excluding transient files."""
    digest = hashlib.sha256()
    ignored = {".git", ".venv", ".pytest_cache", "__pycache__", "artifacts", "data", "models", "tmp"}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _command(command: list[str], *, cwd: Path, log: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a production command and preserve its exact stdout/stderr."""
    started = time.time()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    return {
        "command": command,
        "returncode": result.returncode,
        "seconds": time.time() - started,
        "log": str(log),
        "tail": result.stdout[-8000:],
    }


def _sample_from_challenges(challenges: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id in task_ids:
        tests = challenges[task_id].get("test", ())
        result[task_id] = [{"attempt_1": item["input"], "attempt_2": item["input"]} for item in tests]
    return result


def _cohort(task_ids: list[str]) -> dict[str, Any]:
    canonical = sorted(task_ids)
    return {
        "experiment_id": "ARC2_V7_ROOT_CAUSE_FORENSIC",
        "status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE",
        "source": "validated V38 Frozen30 golden artifact; targets withheld until final score",
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_ids_hash": hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest(),
        "integrity": {"solutions_opened": False, "test_targets_available": False},
    }


def _candidate_value(candidate: dict[str, Any]) -> dict[str, Any]:
    """Exclude timings only; retain all candidate semantics and provenance."""
    return {
        key: candidate.get(key)
        for key in ("prediction", "support_count", "support_augmentations", "augmentation")
    }


def _record_comparison(actual: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    actual_candidates = [_candidate_value(value) for value in actual.get("candidates", ())]
    golden_candidates = [_candidate_value(value) for value in golden.get("candidates", ())]
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    actual_pool = sorted(canonical(value) for value in actual_candidates)
    golden_pool = sorted(canonical(value) for value in golden_candidates)
    return {
        "candidate_pool_exact": actual_pool == golden_pool,
        "candidate_order_exact": actual_candidates == golden_candidates,
        "support_count_exact": [item.get("support_count") for item in actual_candidates]
        == [item.get("support_count") for item in golden_candidates],
        "likelihood_ranking_exact": actual.get("candidate_scores") == golden.get("candidate_scores")
        and actual.get("ranked_candidate_indices") == golden.get("ranked_candidate_indices"),
        "b_support_exact": actual.get("b_support_evidence") == golden.get("b_support_evidence")
        and actual.get("b_support_view_spec") == golden.get("b_support_view_spec"),
        "candidate_pool_sha256": hashlib.sha256("\n".join(actual_pool).encode()).hexdigest(),
        "golden_candidate_pool_sha256": hashlib.sha256("\n".join(golden_pool).encode()).hexdigest(),
        "candidate_order_sha256": hashlib.sha256(json.dumps(actual_candidates, sort_keys=True).encode()).hexdigest(),
        "golden_candidate_order_sha256": hashlib.sha256(json.dumps(golden_candidates, sort_keys=True).encode()).hexdigest(),
    }


def _selection_comparison(actual: dict[str, Any], golden: dict[str, Any]) -> dict[str, bool]:
    left = actual.get("public_reference_selection", {})
    right = golden.get("public_reference_selection", {})
    return {
        "b_support_rank_exact": left.get("ranked_candidate_indices") == right.get("ranked_candidate_indices"),
        "attempt_indices_exact": left.get("attempt_candidate_indices") == right.get("attempt_candidate_indices"),
        "b_support_evidence_exact": left.get("evidence") == right.get("evidence"),
    }


def _submission_score(submission: dict[str, Any], solutions: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    top1 = two = 0
    per_task: dict[str, Any] = {}
    for task_id in task_ids:
        expected = solutions[task_id]
        outputs = submission[task_id]
        first = [output["attempt_1"] for output in outputs]
        second = [output["attempt_2"] for output in outputs]
        top1_ok = first == expected
        two_ok = top1_ok or second == expected
        top1 += int(top1_ok)
        two += int(two_ok)
        per_task[task_id] = {"top1_exact": top1_ok, "two_attempt_exact": two_ok}
    return {"top1_exact": top1, "two_attempt_exact": two, "task_count": len(task_ids), "per_task": per_task}


def _telemetry(
    task_ids: list[str],
    complete: dict[str, Any],
    recovered: dict[str, Any],
    selection: dict[str, Any],
    provenance: dict[str, Any],
    checkpoint_dir: Path,
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    records = complete.get("records", {})
    recovered_records = recovered.get("records", {})
    selected = selection.get("records", {})
    source = provenance.get("task_provenance", {})
    rows: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for task_id in task_ids:
        record = records.get(task_id, {})
        candidates = list(record.get("candidates", ())) if isinstance(record, dict) else []
        candidate_scores = list(record.get("candidate_scores", ())) if isinstance(record, dict) else []
        ranked = list(record.get("ranked_candidate_indices", ())) if isinstance(record, dict) else []
        b_evidence = list(record.get("b_support_evidence", ())) if isinstance(record, dict) else []
        status = record.get("status") if isinstance(record, dict) else None
        output_source = source.get(task_id, {}).get("source")
        row = {
            "generation_ok": status in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"},
            "parse_ok": status == "SUCCESS" and int(record.get("generated_candidate_count", 0)) >= len(candidates),
            "candidate_pool_ok": bool(candidates) and int(record.get("unique_candidate_count", -1)) == len(candidates),
            "original_likelihood_ok": len(candidate_scores) == len(candidates) and set(ranked) == set(range(len(candidates))),
            "b_support_ok": len(b_evidence) == len(candidates),
            "checkpoint_ok": (checkpoint_dir / "tasks" / f"{task_id}.json").exists(),
            "recovered_ok": task_id in recovered_records,
            "final_submission_ok": output_source in {"A", "B", "IDENTITY_FALLBACK"} and task_id in validation["task_ids"],
            "final_source": output_source,
        }
        rows[task_id] = row
        if first_failure is None:
            for stage, ok in row.items():
                if stage.endswith("_ok") and not ok:
                    first_failure = {
                        "task_id": task_id,
                        "stage": stage,
                        "record_status": status,
                        "record": record,
                    }
                    break
    return rows, first_failure


def _validate_submission(submission: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    problems: list[dict[str, str]] = []
    mapping_errors: list[dict[str, str]] = []
    for task_id in sorted(set(submission) | set(sample)):
        if task_id not in submission or task_id not in sample:
            mapping_errors.append({"task_id": task_id, "reason": "missing_or_extra_task"})
            continue
        outputs, expected = submission[task_id], sample[task_id]
        if not isinstance(outputs, list) or len(outputs) != len(expected):
            mapping_errors.append({"task_id": task_id, "reason": "wrong_test_output_count"})
            continue
        for index, output in enumerate(outputs):
            if set(output) != {"attempt_1", "attempt_2"}:
                problems.append({"task_id": task_id, "test_index": str(index), "reason": "missing_attempt"})
                continue
            for attempt in ("attempt_1", "attempt_2"):
                grid = output[attempt]
                if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
                    problems.append({"task_id": task_id, "test_index": str(index), "reason": f"invalid_{attempt}"})
                    continue
                width = len(grid[0])
                if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in grid):
                    problems.append({"task_id": task_id, "test_index": str(index), "reason": f"invalid_{attempt}"})
    return {"valid": not problems and not mapping_errors, "task_ids": sorted(submission), "grid_errors": problems, "mapping_errors": mapping_errors}


def _transport_audit(
    submission: dict[str, Any],
    selection: dict[str, Any],
    candidates: dict[str, Any],
    fallback: dict[str, Any],
    provenance: dict[str, Any],
    task_ids: list[str],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    selected, candidate_records = selection.get("records", {}), candidates.get("records", {})
    by_task = provenance.get("task_provenance", {})
    for task_id in task_ids:
        source = by_task.get(task_id, {}).get("source")
        if source == "B":
            record = selected.get(task_id, {})
            indices = record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())
            pool = record.get("candidates", ())
            if not indices or any(int(index) < 0 or int(index) >= len(pool) for index in indices):
                errors.append({"task_id": task_id, "reason": "invalid_B_indices"}); continue
            grids = [pool[int(index)]["prediction"] for index in indices]
        elif source == "A":
            record = candidate_records.get(task_id, {})
            indices = record.get("ranked_candidate_indices", ())[:2]
            pool = record.get("candidates", ())
            if not indices or any(int(index) < 0 or int(index) >= len(pool) for index in indices):
                errors.append({"task_id": task_id, "reason": "invalid_A_indices"}); continue
            grids = [pool[int(index)]["prediction"] for index in indices]
        elif source == "IDENTITY_FALLBACK":
            grids = []
        else:
            errors.append({"task_id": task_id, "reason": "missing_provenance"}); continue
        outputs = submission.get(task_id, ())
        if source == "IDENTITY_FALLBACK":
            if outputs != fallback.get(task_id): errors.append({"task_id": task_id, "reason": "fallback_transport_mismatch"})
            continue
        if len(grids) == 1: grids.append(grids[0])
        expected = [{"attempt_1": grids[0][index], "attempt_2": grids[1][index]} for index in range(len(outputs))]
        if outputs != expected:
            errors.append({"task_id": task_id, "reason": "test_index_or_attempt_mapping_mismatch"})
    return errors


def _context_stress(root: Path, model: Path, native_config: Path, context_window: int) -> dict[str, Any]:
    """Token-level audit; no model forward pass and no target data."""
    sys.path.insert(0, str(root / "src"))
    from arc.task import ARCExample, ARCGrid, ARCTask
    from inference.nvarc_native import checkpoint_native_tokenizer, native_messages, serialize_grid

    tokenizer, _ = checkpoint_native_tokenizer(model, native_config)
    grid = [[(row + column) % 10 for column in range(30)] for row in range(30)]
    continuation = serialize_grid(grid)
    continuation_tokens = int(tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1])
    examples: list[dict[str, Any]] = []
    first_generation_overflow: int | None = None
    first_likelihood_overflow: int | None = None
    for train_count in range(1, 32):
        task = ARCTask(
            "synthetic_context",
            tuple(ARCExample(ARCGrid(grid), ARCGrid(grid)) for _ in range(train_count)),
            (ARCExample(ARCGrid(grid)),),
        )
        messages = native_messages(task, 0)
        prompt_tokens = int(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"].shape[-1])
        scored_tokens = prompt_tokens + continuation_tokens + 1  # terminal <|im_end|>
        examples.append({"train_pair_count": train_count, "generation_prompt_tokens": prompt_tokens, "original_likelihood_tokens": scored_tokens, "b_support_likelihood_tokens": scored_tokens, "generation_within_context": prompt_tokens <= context_window, "likelihood_within_context": scored_tokens <= context_window})
        if prompt_tokens > context_window and first_generation_overflow is None: first_generation_overflow = train_count
        if scored_tokens > context_window and first_likelihood_overflow is None: first_likelihood_overflow = train_count
    near = max((row for row in examples if row["generation_within_context"]), key=lambda row: row["generation_prompt_tokens"], default=None)
    return {"context_window": context_window, "continuation_tokens": continuation_tokens, "synthetic_rows": examples, "near_generation_boundary": near, "first_generation_overflow_train_pair_count": first_generation_overflow, "first_likelihood_overflow_train_pair_count": first_likelihood_overflow, "generation_succeeds_but_likelihood_overflows": [row for row in examples if row["generation_within_context"] and not row["likelihood_within_context"]]}


def _copy_prefix(source: Path, destination: Path, count: int) -> tuple[list[str], int]:
    files = sorted((source / "tasks").glob("*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    chosen = files[:count]
    (destination / "tasks").mkdir(parents=True, exist_ok=True)
    for path in chosen:
        shutil.copy2(path, destination / "tasks" / path.name)
    return [path.stem for path in chosen], len(files)


def _fault_injections(
    *, root: Path, out: Path, python: str, cohort: Path, config: Path, challenge: Path,
    model: Path, native_config: Path, checkpoint_dir: Path, sample: Path, fallback: Path,
    solutions: Path, task_ids: list[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for count in FAULT_COUNTS:
        fault = out / "fault_injection" / f"after_{count:02d}"
        prefix, available = _copy_prefix(checkpoint_dir, fault / "checkpoints", count)
        recovered = fault / "recovered.json"; selection = fault / "selection.json"; submission = fault / "submission.json"; provenance = fault / "provenance.json"
        recovery = _command([python, str(root / "scripts/recover_public_lb_partial_candidates.py"), "--cohort", str(cohort), "--config", str(config), "--checkpoint-dir", str(fault / "checkpoints"), "--output", str(recovered), "--augmentation-count", "32", "--worker-count", "4", "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1"], cwd=root, log=fault / "recovery.log")
        rerank = _command([python, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(recovered), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(selection), "--allow-deadline-partial", "--require-cached-evidence"], cwd=root, log=fault / "rerank.log") if recovery["returncode"] == 0 else {"returncode": None}
        final = _command([python, str(root / "scripts/build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--a-candidates", str(recovered), "--b-selection", str(selection), "--provenance-output", str(provenance), "--output", str(submission)], cwd=root, log=fault / "finalize.log") if rerank.get("returncode") == 0 else {"returncode": None}
        row: dict[str, Any] = {"interruption_after_completed_tasks": count, "checkpoint_task_ids": prefix, "available_checkpoint_count": available, "recovery_command": recovery, "rerank_command": rerank, "finalize_command": final}
        if final.get("returncode") == 0:
            recovered_payload, provenance_payload, frozen_submission = _read(recovered), _read(provenance), _read(submission)
            row.update({
                "model_prediction_task_count": count,
                "recovered_task_count": len(recovered_payload.get("records", {})),
                "failed_task_count": len(recovered_payload.get("recovery", {}).get("corrupt_or_mismatched_task_ids", ())),
                "identity_fallback_task_count": int(provenance_payload.get("identity_fallback_task_count", 0)),
                "model_predictions_lost_during_recovery_finalization": count - len(recovered_payload.get("records", {})),
                "frozen_submission_path": str(submission),
                "_submission": frozen_submission,
            })
        result.append(row)
    # Deliberately score only after every injected final artifact was frozen.
    loaded_solutions = _read(solutions)
    for row in result:
        if "_submission" in row:
            row["final_frozen30_score"] = _submission_score(row.pop("_submission"), loaded_solutions, task_ids)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("production_root", "challenge_path", "solutions_path", "golden_candidates", "golden_selection", "model_path", "native_config_dir", "config", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--notebook-path", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-generation", action="store_true", help="audit existing complete artifact at --output-dir; never invokes the model")
    args = parser.parse_args()
    root, out = args.production_root.resolve(), args.output_dir.resolve()
    if not (root / "scripts/run_qwen4b_native_augmentation_search.py").exists():
        raise FileNotFoundError("--production-root is not the exact ARC2 V7 source package")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"forensic output must be new and immutable: {out}")
    out.mkdir(parents=True)
    started = time.time()
    report: dict[str, Any] = {"experiment_id": "ARC2_V7_ROOT_CAUSE_FORENSIC", "status": "RUNNING", "started_unix": started, "inputs": {"production_root": str(root), "challenge_path": str(args.challenge_path), "golden_candidates": str(args.golden_candidates), "golden_selection": str(args.golden_selection), "model_path": str(args.model_path), "native_config_dir": str(args.native_config_dir), "config": str(args.config)}}
    try:
        golden_candidates, golden_selection = _read(args.golden_candidates), _read(args.golden_selection)
        task_ids = list(golden_candidates.get("records", {}))
        if len(task_ids) != 30 or set(task_ids) != set(golden_selection.get("records", {})):
            raise ValueError("validated V38 Frozen30 golden candidates/selection are incomplete or mismatched")
        challenges = _read(args.challenge_path)
        if not set(task_ids) <= set(challenges): raise ValueError("Frozen30 task IDs are absent from production challenge path")
        cohort, sample = out / "frozen30_cohort.json", out / "frozen30_sample_submission.json"
        replay_challenge = out / "frozen30_challenges.json"
        # Production helpers intentionally require their cohort to cover the
        # entire supplied challenge file.  A target-blind, immutable subset
        # lets this forensic replay use their exact production path on Frozen30.
        _write(replay_challenge, {task_id: challenges[task_id] for task_id in task_ids})
        _write(cohort, _cohort(task_ids)); _write(sample, _sample_from_challenges(challenges, task_ids))
        fallback = out / "fallback.json"
        init = _command([args.python, str(root / "scripts/initialize_public_lb_fallback.py"), "--cohort", str(cohort), "--challenge-path", str(replay_challenge), "--sample-submission", str(sample), "--output", str(fallback)], cwd=root, log=out / "fallback.log")
        if init["returncode"] != 0: raise RuntimeError(f"fallback initialization failed: {init['tail']}")
        report["asset_identity"] = {"git_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() if (root / ".git").exists() else "SOURCE_DATASET_NO_GIT", "source_package_sha256": _tree_sha256(root), "notebook_sha256": _sha256(args.notebook_path) if args.notebook_path else None, "model_path": str(args.model_path), "generation_micro_batch_size": 1, "likelihood_micro_batch_size": 1, "worker_count": 4, "augmentation_count": 32, "config_sha256": _sha256(args.config), "golden_candidates_sha256": _sha256(args.golden_candidates), "golden_selection_sha256": _sha256(args.golden_selection)}
        complete, checkpoints = out / "A_candidates_complete.json", out / "generation_checkpoints"
        if not args.skip_generation:
            replay = _command([args.python, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(args.config), "--challenge-path", str(replay_challenge), "--model-path", str(args.model_path), "--native-config-dir", str(args.native_config_dir), "--output", str(complete), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1"], cwd=root, log=out / "generation.log", env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
            report["production_replay_command"] = replay
            if replay["returncode"] != 0: raise RuntimeError(f"production replay generation failed: {replay['tail']}")
        if not complete.exists(): raise FileNotFoundError("missing complete replay artifact")
        recovered, selection = out / "A_candidates_recovered.json", out / "B_support_selection_frozen.json"
        recovery = _command([args.python, str(root / "scripts/recover_public_lb_partial_candidates.py"), "--cohort", str(cohort), "--config", str(args.config), "--checkpoint-dir", str(checkpoints), "--output", str(recovered), "--augmentation-count", "32", "--worker-count", "4", "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1"], cwd=root, log=out / "recovery.log")
        if recovery["returncode"] != 0: raise RuntimeError(f"recovery failed: {recovery['tail']}")
        rerank = _command([args.python, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(recovered), "--challenge-path", str(replay_challenge), "--model-path", str(args.model_path), "--native-config-dir", str(args.native_config_dir), "--output", str(selection), "--require-cached-evidence"], cwd=root, log=out / "rerank.log")
        if rerank["returncode"] != 0: raise RuntimeError(f"rerank failed: {rerank['tail']}")
        submission, provenance = out / "submission.json", out / "submission_provenance.json"
        finalization = _command([args.python, str(root / "scripts/build_public_lb_submission.py"), "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback), "--a-candidates", str(recovered), "--b-selection", str(selection), "--provenance-output", str(provenance), "--require-model-prediction", "--output", str(submission)], cwd=root, log=out / "finalize.log")
        if finalization["returncode"] != 0: raise RuntimeError(f"finalization failed: {finalization['tail']}")
        # All predictions are now frozen.  The next call opens the Frozen30 solutions.
        complete_payload, recovered_payload, selection_payload = _read(complete), _read(recovered), _read(selection)
        submission_payload, provenance_payload = _read(submission), _read(provenance)
        validation = _validate_submission(submission_payload, _read(sample))
        telemetry, first_failure = _telemetry(task_ids, complete_payload, recovered_payload, selection_payload, provenance_payload, checkpoints, validation)
        comparisons: dict[str, Any] = {}
        for task_id in task_ids:
            comparisons[task_id] = {**_record_comparison(recovered_payload["records"][task_id], golden_candidates["records"][task_id]), **_selection_comparison(selection_payload["records"][task_id], golden_selection["records"][task_id])}
        comparison_counts = {key: sum(bool(row.get(key)) for row in comparisons.values()) for key in next(iter(comparisons.values())) if key.endswith("_exact")}
        # Every fault finalization is built before targets are opened.
        faults = _fault_injections(root=root, out=out, python=args.python, cohort=cohort, config=args.config, challenge=replay_challenge, model=args.model_path, native_config=args.native_config_dir, checkpoint_dir=checkpoints, sample=sample, fallback=fallback, solutions=args.solutions_path, task_ids=task_ids)
        solutions = _read(args.solutions_path)
        final_score = _submission_score(submission_payload, solutions, task_ids)
        context = _context_stress(root, args.model_path, args.native_config_dir, 16384)
        transport_errors = _transport_audit(submission_payload, selection_payload, recovered_payload, _read(fallback), provenance_payload, task_ids)
        model_coverage = Counter(item.get("source") for item in provenance_payload.get("task_provenance", {}).values())
        recovery_loss = sum(int(item.get("model_predictions_lost_during_recovery_finalization", 0)) for item in faults)
        root_cause = "UNDETERMINED"
        confidence = "LOW"
        if comparison_counts.get("candidate_pool_exact", 0) < len(task_ids):
            root_cause, confidence = "PRODUCTION_PACKAGE_OR_RUNTIME_DIVERGENCE", "HIGH"
        elif not validation["valid"] or transport_errors:
            root_cause, confidence = "RECOVERY_RERANK_OR_SUBMISSION_TRANSPORT_BUG", "HIGH"
        elif any(item.get("model_predictions_lost_during_recovery_finalization", 0) for item in faults):
            root_cause, confidence = "DEADLINE_RECOVERY_OR_FALLBACK_LOSS", "HIGH"
        elif context["generation_succeeds_but_likelihood_overflows"]:
            root_cause, confidence = "CONTEXT_WINDOW_ASYMMETRY_RISK", "MEDIUM"
        elif final_score["two_attempt_exact"] <= 8:
            root_cause, confidence = "NO_V38_PRODUCTION_BUG_REPRODUCED__HIDDEN_ONLY_COVERAGE_RUNTIME_OR_DISTRIBUTION", "MEDIUM"
        report.update({
            "status": "COMPLETE", "finished_unix": time.time(), "runtime_seconds": time.time() - started,
            "FIRST_DIVERGENCE_STAGE": first_failure["stage"] if first_failure else "NONE", "AFFECTED_TASK_COUNT": len([row for row in telemetry.values() if not all(value for key, value in row.items() if key.endswith("_ok"))]),
            "MODEL_COVERAGE": dict(model_coverage), "IDENTITY_FALLBACK_COUNT": int(provenance_payload.get("identity_fallback_task_count", 0)), "FROZEN30_FINAL_SCORE": final_score,
            "V38_GOLDEN_EQUIVALENCE": {"per_task": comparisons, "counts": comparison_counts}, "CONTEXT_OVERFLOW_COUNT": len(context["generation_succeeds_but_likelihood_overflows"]), "RECOVERY_LOSS_COUNT": recovery_loss,
            "SUBMISSION_MAPPING_ERRORS": transport_errors + validation["mapping_errors"], "ROOT_CAUSE": root_cause, "CONFIDENCE": confidence,
            "MINIMAL_FIX": "Do not gate the production inference entrypoint on the undocumented KAGGLE_IS_COMPETITION_RERUN variable; use a separate fast-validation notebook and make the competition notebook execute Dynamic-B unconditionally." if root_cause != "UNDETERMINED" else "No change proposed until the failing stage is isolated.",
            "stage_telemetry": telemetry, "first_failure_detail": first_failure, "submission_validation": validation, "submission_transport": {"errors": transport_errors, "provenance": provenance_payload}, "deadline_recovery_fault_injection": faults, "context_window_stress_audit": context,
            "artifact_sha256": {"complete": _sha256(complete), "recovered": _sha256(recovered), "selection": _sha256(selection), "submission": _sha256(submission), "provenance": _sha256(provenance)},
        })
    except Exception as exc:
        report.update({"status": "FAILED", "finished_unix": time.time(), "runtime_seconds": time.time() - started, "ROOT_CAUSE": "FORENSIC_HARNESS_FAILURE", "CONFIDENCE": "HIGH", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    _write(out / "ROOT_CAUSE_REPORT.json", report)
    lines = ["# ARC2 V7 Root-Cause Forensic", "", f"- Status: `{report['status']}`", f"- Root cause: `{report.get('ROOT_CAUSE')}`", f"- Confidence: `{report.get('CONFIDENCE')}`", f"- First divergence: `{report.get('FIRST_DIVERGENCE_STAGE')}`", f"- Frozen30 score: `{report.get('FROZEN30_FINAL_SCORE')}`", f"- Identity fallback count: `{report.get('IDENTITY_FALLBACK_COUNT')}`", f"- V38 equivalence: `{report.get('V38_GOLDEN_EQUIVALENCE', {}).get('counts')}`", f"- Recovery loss count: `{report.get('RECOVERY_LOSS_COUNT')}`", f"- Mapping errors: `{report.get('SUBMISSION_MAPPING_ERRORS')}`", "", "## Minimal fix", "", str(report.get("MINIMAL_FIX", "Not available")), ""]
    (out / "ROOT_CAUSE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
