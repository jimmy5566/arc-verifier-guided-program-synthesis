"""Create the V8 always-production notebook from the frozen V38 builder."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_source(builder: Path) -> str:
    spec = importlib.util.spec_from_file_location("v38_production_builder", builder)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V38 notebook builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module._source())


def _remove_between(source: str, start: str, end: str) -> str:
    left = source.find(start)
    right = source.find(end, left)
    if left < 0 or right < 0:
        raise RuntimeError(f"required V38 source section not found: {start!r}")
    return source[:left] + source[right:]


def _always_production_source(builder: Path) -> str:
    source = _load_source(builder)
    source = _remove_between(source, "environment_audit = _sanitized_environment_audit()", "def validate_submission")
    source = _remove_between(source, "def write_fast_commit_marker():", "def run_with_watchdog")
    old_tail = "if is_competition_rerun:\n    run_competition_rerun()\nelse:\n    write_fast_commit_marker()\n"
    if old_tail not in source:
        raise RuntimeError("V38 rerun-gate tail was not found")
    diagnostics = r'''# V8 has no execution gate: every notebook execution is the production run.
print(json.dumps({"event": "ALWAYS_PRODUCTION_MODE", "rerun_gate_present": False, "fast_commit_present": False, "solutions_opened": False}, sort_keys=True), flush=True)
run_competition_rerun()
'''
    source = source.replace(old_tail, diagnostics)
    anchor = '    print(json.dumps({"event": "ARC_PRIZE_2026_PRODUCTION_COMPLETE", "submission": str(submission), "validation": "PASS", **validation, "fallback_counts": {key: counts[key] for key in ("b_task_count", "a_fallback_task_count", "identity_fallback_task_count")}, "solutions_opened": False, "hard_finalize_remaining_seconds": max(0.0, hard_finalize_unix - time.time())}, sort_keys=True), flush=True)\n'
    if anchor not in source:
        raise RuntimeError("V38 completion anchor was not found")
    summary = r'''    recovered_payload = json.loads(recovered_a.read_text(encoding="utf-8"))
    selection_payload = json.loads(b_selection.read_text(encoding="utf-8")) if b_selection.exists() else {"records": {}}
    recovered_records = recovered_payload.get("records", {})
    selected_records = selection_payload.get("records", {})
    recovery_info = recovered_payload.get("recovery", {})
    completed_task_ids = sorted(recovered_records)
    unfinished_task_ids = sorted(set(challenges) - set(completed_task_ids))
    per_task_manifest = {
        task_id: {
            "checkpoint_recovered": task_id in recovered_records,
            "b_selected": task_id in selected_records,
            "final_source": counts and json.loads(provenance.read_text(encoding="utf-8")).get("task_provenance", {}).get(task_id, {}).get("source"),
        }
        for task_id in sorted(challenges)
    }
    diagnostic_summary = {
        "event": "ARC2_V8_VISIBLE_PRODUCTION_AUDIT",
        "total_tasks": len(challenges),
        "completed_model_tasks": len(completed_task_ids),
        "recovered_tasks": len(recovered_records),
        "B_selected_tasks": int(counts["b_task_count"]),
        "A_fallback_tasks": int(counts["a_fallback_task_count"]),
        "identity_fallback_tasks": int(counts["identity_fallback_task_count"]),
        "failed_tasks": sorted(recovery_info.get("corrupt_or_mismatched_task_ids", ())),
        "unfinished_tasks": unfinished_task_ids,
        "total_runtime_seconds": time.time() - started_unix,
        "submission_task_count": len(json.loads(submission.read_text(encoding="utf-8"))),
        "test_output_count": sum(len(value) for value in json.loads(submission.read_text(encoding="utf-8")).values()),
        "provenance_health": len(completed_task_ids) == len(challenges) and not unfinished_task_ids and int(counts["identity_fallback_task_count"]) == 0,
        "solutions_opened": False,
    }
    def _atomic_diagnostic(name, value):
        path = out / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    _atomic_diagnostic("per_task_completion_manifest.json", per_task_manifest)
    _atomic_diagnostic("checkpoint_recovery_summary.json", recovery_info)
    _atomic_diagnostic("runtime_summary.json", {"total_runtime_seconds": diagnostic_summary["total_runtime_seconds"], "soft_dispatch_cutoff_unix": soft_dispatch_cutoff_unix, "hard_inference_stop_unix": hard_inference_stop_unix, "hard_finalize_unix": hard_finalize_unix})
    _atomic_diagnostic("production_diagnostic_summary.json", diagnostic_summary)
    print(json.dumps(diagnostic_summary, sort_keys=True), flush=True)
    if diagnostic_summary["completed_model_tasks"] < diagnostic_summary["total_tasks"]:
        raise RuntimeError("V8 unhealthy production run: fewer than 240 tasks received model execution")
'''
    source = source.replace(anchor, anchor + summary)
    if "KAGGLE_IS_COMPETITION_RERUN" in source or "FAST_COMMIT_MODE" in source or "write_fast_commit_marker" in source:
        raise RuntimeError("V8 source still contains forbidden rerun-gate/fast-commit code")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = _always_production_source(args.base_builder)
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
