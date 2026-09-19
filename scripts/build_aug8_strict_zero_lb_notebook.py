"""Build the always-production Aug8 strict-coverage Kaggle diagnostic."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_base(builder: Path) -> str:
    spec = importlib.util.spec_from_file_location("v38_production_builder", builder)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V38 production notebook builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module._source())


def _remove_between(source: str, start: str, end: str) -> str:
    left = source.find(start)
    right = source.find(end, left)
    if left < 0 or right < 0:
        raise RuntimeError(f"required notebook section was not found: {start!r}")
    return source[:left] + source[right:]


def _replace_exact(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one {label} section, found {source.count(old)}")
    return source.replace(old, new)


def _source(base_builder: Path) -> str:
    source = _load_base(base_builder)
    # V38's optional rerun gate and fast-commit marker are removed entirely.
    source = _remove_between(source, "environment_audit = _sanitized_environment_audit()", "def validate_submission")
    source = _remove_between(source, "def write_fast_commit_marker():", "def run_with_watchdog")
    source = _replace_exact(
        source,
        "if is_competition_rerun:\n    run_competition_rerun()\nelse:\n    write_fast_commit_marker()\n",
        "print(json.dumps({\"event\": \"AUG8_ALWAYS_PRODUCTION_MODE\", \"rerun_gate_present\": False, \"fast_commit_present\": False, \"solutions_opened\": False}, sort_keys=True), flush=True)\nrun_competition_rerun()\n",
        "rerun-gate tail",
    )
    source = _replace_exact(source, '    fallback = out / "submission_fallback.json"\n', "", "fallback declaration")
    source = _remove_between(source, "    if not fallback.exists():", "    if time.time() < soft_dispatch_cutoff_unix:")
    source = _replace_exact(source, '"--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4",', '"--stage", "external", "--external-augmentation-count", "8", "--external-worker-count", "4",', "generation augmentation count")
    source = _replace_exact(source, '"--deadline-unix", str(soft_dispatch_cutoff_unix), "--allow-deadline-partial",', '"--deadline-unix", str(soft_dispatch_cutoff_unix),', "permissive generation deadline")
    source = _replace_exact(source, '"--output", str(recovered_a), "--augmentation-count", "32", "--worker-count", "4",', '"--output", str(recovered_a), "--augmentation-count", "8", "--worker-count", "4",', "recovery augmentation count")
    source = _replace_exact(source, '"--deadline-unix", str(hard_finalize_unix), "--allow-deadline-partial", "--require-cached-evidence",', '"--deadline-unix", str(hard_finalize_unix), "--require-cached-evidence",', "permissive B deadline")

    old_recovery = '    if recovery["returncode"]: raise RuntimeError("RECOVER_VALID_A_CHECKPOINTS failed")\n'
    strict_recovery = r'''    recovery_payload = json.loads(recovered_a.read_text(encoding="utf-8")) if recovered_a.exists() else {"records": {}, "recovery": {}}
    recovered_records = recovery_payload.get("records", {}) if isinstance(recovery_payload.get("records", {}), dict) else {}
    recovery_info = recovery_payload.get("recovery", {}) if isinstance(recovery_payload.get("recovery", {}), dict) else {}
    strict_preselection = {
        "expected_tasks": len(challenges),
        "recovered_model_tasks": len(recovered_records),
        "failed_tasks": len(recovery_info.get("corrupt_or_mismatched_task_ids", ())),
        "unfinished_tasks": len(set(challenges) - set(recovered_records)),
        "generation_returncode": generation.get("returncode") if "generation" in locals() else None,
        "recovery_returncode": recovery.get("returncode"),
        "augmentation_count": 8,
        "solutions_opened": False,
    }
    def _atomic_json(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    _atomic_json(out / "strict_preselection_diagnostic.json", strict_preselection)
    if (
        recovery.get("returncode")
        or strict_preselection["generation_returncode"] not in (0, None)
        or strict_preselection["expected_tasks"] != 240
        or strict_preselection["recovered_model_tasks"] != 240
        or strict_preselection["failed_tasks"] != 0
        or strict_preselection["unfinished_tasks"] != 0
        or set(recovered_records) != set(challenges)
    ):
        print(json.dumps({"event": "STRICT_HIDDEN_COVERAGE_FAILURE", "phase": "recovery", **strict_preselection}, sort_keys=True), flush=True)
        raise RuntimeError("STRICT_HIDDEN_COVERAGE_FAILED")
'''
    source = _replace_exact(source, old_recovery, strict_recovery, "recovery validation")

    old_selection_result = '        print(json.dumps({"event": "FROZEN_B_SUPPORT_SELECTION_RESULT", **selection}, sort_keys=True), flush=True)\n'
    strict_selection = r'''        print(json.dumps({"event": "FROZEN_B_SUPPORT_SELECTION_RESULT", **selection}, sort_keys=True), flush=True)
    selection_payload = json.loads(b_selection.read_text(encoding="utf-8")) if b_selection.exists() else {"records": {}}
    selected_records = selection_payload.get("records", {}) if isinstance(selection_payload.get("records", {}), dict) else {}
    strict_selection_diagnostic = {
        "expected_tasks": len(challenges), "B_selected_tasks": len(selected_records),
        "selection_returncode": selection.get("returncode") if "selection" in locals() else None,
        "selection_status": selection_payload.get("status"), "solutions_opened": False,
    }
    _atomic_json(out / "strict_selection_diagnostic.json", strict_selection_diagnostic)
    if (
        strict_selection_diagnostic["expected_tasks"] != 240
        or strict_selection_diagnostic["B_selected_tasks"] != 240
        or strict_selection_diagnostic["selection_returncode"] not in (0, None)
        or set(selected_records) != set(challenges)
    ):
        print(json.dumps({"event": "STRICT_HIDDEN_COVERAGE_FAILURE", "phase": "selection", **strict_selection_diagnostic}, sort_keys=True), flush=True)
        raise RuntimeError("STRICT_HIDDEN_COVERAGE_FAILED")
'''
    source = _replace_exact(source, old_selection_result, strict_selection, "selection validation")

    old_final = r'''    finalization = run_with_watchdog([
        sys.executable, str(root / "scripts/build_public_lb_submission.py"),
        "--cohort", str(cohort), "--sample-submission", str(sample), "--fallback", str(fallback),
        "--a-candidates", str(recovered_a), "--b-selection", str(b_selection),
        "--provenance-output", str(provenance), "--require-model-prediction", "--output", str(submission),
    ], "SUBMISSION_FINALIZE", hard_finalize_unix)
    if finalization["returncode"]:
        raise RuntimeError("SUBMISSION_FINALIZE failed: production produced no model predictions or invalid artifact")
    validation = validate_submission(submission, sample_submission)
    counts = json.loads(provenance.read_text(encoding="utf-8"))
    if counts["identity_fallback_task_count"] == len(challenges):
        raise RuntimeError("production validation failed: all tasks used identity fallback")
    print(json.dumps({"event": "ARC_PRIZE_2026_PRODUCTION_COMPLETE", "submission": str(submission), "validation": "PASS", **validation, "fallback_counts": {key: counts[key] for key in ("b_task_count", "a_fallback_task_count", "identity_fallback_task_count")}, "solutions_opened": False, "hard_finalize_remaining_seconds": max(0.0, hard_finalize_unix - time.time())}, sort_keys=True), flush=True)
'''
    strict_final = r'''    finalization = run_with_watchdog([
        sys.executable, str(root / "scripts/build_strict_public_lb_submission.py"),
        "--cohort", str(cohort), "--sample-submission", str(sample),
        "--a-candidates", str(recovered_a), "--b-selection", str(b_selection),
        "--provenance-output", str(provenance), "--output", str(submission),
    ], "STRICT_SUBMISSION_FINALIZE", hard_finalize_unix)
    if finalization["returncode"]:
        raise RuntimeError("STRICT_HIDDEN_COVERAGE_FAILED")
    validation = validate_submission(submission, sample_submission)
    strict_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    counts = strict_provenance.get("counts", {})
    if counts != {
        "expected_task_count": 240, "recovered_model_task_count": 240,
        "b_task_count": 240, "a_fallback_task_count": 0,
        "identity_fallback_task_count": 0, "failed_task_count": 0,
        "unfinished_task_count": 0, "submission_task_count": 240,
        "submission_test_output_count": sum(len(value) for value in sample_submission.values()),
    }:
        print(json.dumps({"event": "STRICT_HIDDEN_COVERAGE_FAILURE", "phase": "finalization", "counts": counts}, sort_keys=True), flush=True)
        raise RuntimeError("STRICT_HIDDEN_COVERAGE_FAILED")
    per_task = strict_provenance.get("per_task", {})
    _atomic_json(out / "strict_per_task_manifest.json", per_task)
    _atomic_json(out / "strict_runtime_summary.json", {
        "total_runtime_seconds": time.time() - started_unix,
        "soft_dispatch_cutoff_unix": soft_dispatch_cutoff_unix,
        "hard_inference_stop_unix": hard_inference_stop_unix,
        "hard_finalize_unix": hard_finalize_unix,
        "generation": generation,
        "recovery": recovery,
        "selection": selection,
        "finalization": finalization,
    })
    print(json.dumps({"event": "AUG8_STRICT_PRODUCTION_COMPLETE", "submission": str(submission), "validation": "PASS", **validation, "strict_counts": counts, "submission_sha256": strict_provenance["submission_sha256"], "solutions_opened": False, "hard_finalize_remaining_seconds": max(0.0, hard_finalize_unix - time.time())}, sort_keys=True), flush=True)
'''
    source = _replace_exact(source, old_final, strict_final, "finalization block")
    forbidden = ("KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT_MODE", "write_fast_commit_marker", '"--external-augmentation-count", "32"', '"--augmentation-count", "32"', "--allow-deadline-partial")
    present = [item for item in forbidden if item in source]
    if present:
        raise RuntimeError(f"strict Aug8 source retains forbidden production behavior: {present}")
    for required in ("build_strict_public_lb_submission.py", "STRICT_HIDDEN_COVERAGE_FAILED", '"--external-augmentation-count", "8"', "AUG8_ALWAYS_PRODUCTION_MODE"):
        if required not in source:
            raise RuntimeError(f"strict Aug8 source lacks required contract: {required}")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = _source(args.base_builder)
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
