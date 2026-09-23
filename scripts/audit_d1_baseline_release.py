#!/usr/bin/env python3
"""CPU-only release-readiness audit for the frozen D1 review baseline.

It intentionally reports unmet production contracts as ``RELEASE_BLOCKED``.
It neither loads a model nor writes a Kaggle artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "release" / "TTT24_TTT48_4PLUS4_D1_BASELINE_V1"
DEFAULT_D1 = ROOT / "artifacts" / "eval60_4plus4_d1_selector"
DEFAULT_LOG = ROOT / "artifacts" / "kaggle_downloads" / "ttt48_missing_cross_scores_v2" / "version2_logs.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _cross_score_anchor_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "NOT_AVAILABLE", "anchor_mismatch_task_count": None, "completed_task_count": None}
    events = _read(path)
    text = "\n".join(str(row.get("data", "")) for row in events if isinstance(row, dict))
    mismatches = sorted(set(re.findall(r"TTT48 anchor mismatch for ([0-9a-f]+)", text)))
    complete = sorted(set(re.findall(r'"event": "TASK_CROSS_SCORES_FROZEN"[^\n]*?"task_id": "([0-9a-f]+)"', text)))
    final = re.findall(r'TTT48_CROSS_SCORE_RUN_COMPLETE.*?"completed_requests": (\d+).*?"completed_tasks": (\d+).*?"missing_tasks": (\d+)', text)
    return {
        "status": "FAILED_NUMERICAL_PARITY" if mismatches else "NO_ANCHOR_MISMATCH_FOUND",
        "log_sha256": _sha256(path),
        "anchor_mismatch_task_count": len(mismatches),
        "anchor_mismatch_task_ids": mismatches,
        "completed_task_ids_visible_in_log": complete,
        "runner_final_event": final[-1] if final else None,
    }


def _production_static_audit() -> dict[str, Any]:
    """Audit the isolated D1 release route, never the retired Aug8 route."""
    builder = (ROOT / "scripts" / "build_d1_release_kaggle.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_d1_release_4gpu.py").read_text(encoding="utf-8")
    finalizer = (ROOT / "scripts" / "build_d1_release_submission.py").read_text(encoding="utf-8")
    contract = (ROOT / "src" / "inference" / "d1_release_contract.py").read_text(encoding="utf-8")
    issues: list[str] = []
    if "FAST_COMMIT_MODE" in builder or "STRUCTURAL_DUMMY_ONLY" in builder:
        issues.append("RERUN_PATH_NOT_EQUIVALENT: D1 builder retains a fast-commit/dummy branch")
    if "len(task_ids) != 240" in runner or "240" in finalizer:
        issues.append("RUNTIME_IDENTITY_NOT_DYNAMIC: D1 route contains a fixed visible task count")
    if "release_identity" not in runner or "task_contract" not in contract:
        issues.append("CHECKPOINT_IDENTITY_NOT_BOUND_TO_RUNTIME_CHALLENGE_CONTENT_AND_TEST_SHAPE")
    if ".rglob(" in builder:
        issues.append("AMBIGUOUS_INPUT_DISCOVERY: D1 builder uses rglob source discovery")
    if "select_record" not in finalizer or "fixed_TTT24_TTT48_4plus4_per_output_D1" not in contract:
        issues.append("D1_PER_OUTPUT_FINALIZER_NOT_WIRED")
    if "release-image-only" in runner:
        issues.append("LIVE_D1_WORKER_BOOTSTRAP_UNBOUND: CPU route is verified, but the exact CUDA TTT24/48 worker has not been parity-bound")
    return {"status": "FAIL" if issues else "PASS", "issues": issues,
            "builder_sha256": hashlib.sha256(builder.encode()).hexdigest(),
            "runner_sha256": hashlib.sha256(runner.encode()).hexdigest(),
            "finalizer_sha256": hashlib.sha256(finalizer.encode()).hexdigest()}


def _d1_replay_state(d1_dir: Path) -> dict[str, Any]:
    report = _read(d1_dir / "D1_REPLAY_REPORT.json")
    predictions = _read(d1_dir / "d1_predictions_frozen.json")
    baseline = _read(d1_dir / "baseline_replay_frozen.json")
    count = sum(len(value["attempt_1"]) for value in predictions["predictions"].values())
    if (report["d1"]["top1"], report["d1"]["top2"], report["d1"]["pool_oracle"], count) != (20, 28, 30, 89):
        raise ValueError("D1 replay evidence does not match the frozen contract")
    return {
        "status": "PASS",
        "top1": report["d1"]["top1"], "top2": report["d1"]["top2"], "pool_oracle": report["d1"]["pool_oracle"],
        "attempt_record_count": count, "historical_replay_exact": report["integrity"]["historical_rankings_exactly_reproduced"],
        "candidate_sets_unchanged": report["integrity"]["candidate_sets_unchanged"],
        "hashes": {name: _sha256(d1_dir / name) for name in ("d1_config_frozen.json", "baseline_replay_frozen.json", "d1_rankings_frozen.json", "d1_predictions_frozen.json", "D1_REPLAY_REPORT.json")},
        "baseline_replay_sha256": _sha256(d1_dir / "baseline_replay_frozen.json"),
    }


def _markdown(report: dict[str, Any]) -> str:
    blockers = report["release_blockers"]
    lines = [
        "# D1 baseline release review",
        "",
        "This is a CPU-only review gate. It is not a Kaggle run, a hidden-coverage result, or a competition submission.",
        "",
        f"- Baseline: `{report['baseline_name']}`",
        f"- D1 replay: Top-1 `{report['d1_replay']['top1']}/89`; Top-2 `{report['d1_replay']['top2']}/89`; pool oracle `{report['d1_replay']['pool_oracle']}/89`.",
        f"- Remaining Kaggle GPU quota observed: `{report['remaining_quota']}`.",
        f"- CPU test suite: `{report['cpu_test_summary']}`.",
        f"- Release status: **{report['release_status']}**.",
        "",
        "## Blockers",
        *[f"- {item}" for item in blockers],
        "",
        "## Required before any release re-evaluation",
        "",
        "1. Establish the TTT48 numerical anchor parity using the same verified adaptation path; do not relax its tolerance.",
        "2. Bind the verified live CUDA worker bootstrap into the D1 release image, then run a separate no-submission GPU parity gate.",
        "3. Re-evaluate GPU quota only when a cost estimate and recovery reserve are supplied.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1)
    parser.add_argument("--cross-score-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--remaining-quota", type=float)
    parser.add_argument("--estimated-gpu-hours", type=float)
    parser.add_argument("--recovery-reserve-gpu-hours", type=float)
    parser.add_argument("--cpu-test-summary", default="NOT_RUN")
    args = parser.parse_args()
    for filename in ("D1_BASELINE_CONFIG.json",):
        if not (args.release_dir / filename).is_file():
            raise FileNotFoundError(args.release_dir / filename)
    d1 = _d1_replay_state(args.d1_dir)
    production = _production_static_audit()
    anchor = _cross_score_anchor_state(args.cross_score_log)
    from inference.d1_release_contract import quota_status
    quota = quota_status(available=args.remaining_quota, estimate=args.estimated_gpu_hours, reserve=args.recovery_reserve_gpu_hours)
    blockers = []
    blockers.extend(production["issues"])
    if anchor["status"] != "NO_ANCHOR_MISMATCH_FOUND":
        blockers.append("MODEL_STATE_PARITY_UNVERIFIED: failed TTT48 cross-score run had numerical anchor mismatches")
    if quota["status"] == "QUOTA_INSUFFICIENT":
        blockers.append("QUOTA_INSUFFICIENT: supplied cost estimate plus recovery reserve exceeds available quota")
    report = {
        "baseline_name": "TTT24_TTT48_4PLUS4_D1_BASELINE_V1",
        "scope": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE_NOT_HELD_OUT_NOT_LB_PERFORMANCE",
        "d1_replay": d1,
        "live_4plus4_evidence_parity": "NOT_VERIFIED",
        "model_state_parity": anchor,
        "rerun_path_test": "FAIL_STATIC" if production["status"] != "PASS" else "NOT_RUN",
        "failure_injection_tests": "CPU_D1_RELEASE_CONTRACT_TESTS_REQUIRED",
        "cpu_test_summary": args.cpu_test_summary,
        "gpu_smoke_status": "NOT_RUN_QUOTA_AND_MODEL_STATE_BLOCKED",
        "full_saved_run_status": "NOT_RUN_QUOTA_AND_MODEL_STATE_BLOCKED",
        "expected_tasks_completed_tasks": "NOT_RUN",
        "expected_outputs_valid_outputs": "NOT_RUN",
        "empty_pool_count": "NOT_RUN",
        "fallback_counts_by_reason": "NOT_RUN",
        "failed_unfinished": "NOT_RUN",
        "total_runtime": "NOT_RUN",
        "quota_status": quota,
        "remaining_quota": args.remaining_quota,
        "submission_sha256": "NOT_CREATED",
        "production_static_audit": production,
        "source_commit": _git(["git", "rev-parse", "HEAD"]),
        "release_blockers": blockers,
        "release_status": "RELEASE_BLOCKED",
        "competition_submitted": "NO",
    }
    _write(args.release_dir / "RELEASE_STATUS.json", report)
    (args.release_dir / "RELEASE_STATUS.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"event": "D1_RELEASE_REVIEW_COMPLETE", "release_status": report["release_status"], "blocker_count": len(blockers), "top2": d1["top2"]}, sort_keys=True))


if __name__ == "__main__":
    main()
