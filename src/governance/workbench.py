"""Thin, model-free governance layer around the existing fixed-4+4+D1 solver.

This module owns configuration resolution, target separation, run state and
artifact identity.  It deliberately does not implement model loading, TTT,
generation, likelihood scoring, D1 selection, or Kaggle publishing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from inference.d1_release_contract import PORTFOLIO, validate_grid


SCHEMA_VERSION = "ARC2_GOVERNANCE_V1"
RUN_STATES = {"PREPARED", "RUNNING", "COMPLETE_FROZEN", "COMPLETE_SCORED", "INCOMPLETE", "FAILED"}
FALSE_VALUES = {"", "0", "false", "no"}
TRUE_VALUES = {"1", "true", "yes"}


class GovernanceError(ValueError):
    """Configuration, provenance, or state violates the governance contract."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=sort_keys)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(dict(event)) + "\n")


def parse_rerun_flag(raw: str | None) -> bool:
    value = (raw or "").strip().lower()
    if value in FALSE_VALUES:
        return False
    if value in TRUE_VALUES:
        return True
    raise GovernanceError(f"unexpected KAGGLE_IS_COMPETITION_RERUN value: {raw!r}")


def assert_experiment_phase(environ: Mapping[str, str] | None = None) -> None:
    values = os.environ if environ is None else environ
    if parse_rerun_flag(values.get("KAGGLE_IS_COMPETITION_RERUN")):
        raise GovernanceError("experiment workbench rejects competition rerun execution")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernanceError(f"{name} must be an object")
    return value


def _require(mapping: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise GovernanceError(f"{name} missing required fields: {missing}")


def resolve_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all scientific settings explicitly; introduce no defaults."""
    _require(config, {"algorithm", "experiment", "execution"}, "config")
    algorithm = _mapping(config["algorithm"], "algorithm")
    experiment = _mapping(config["experiment"], "experiment")
    execution = _mapping(config["execution"], "execution")
    _require(
        algorithm,
        {"solver_id", "source_commit", "source_package_sha256", "release_config", "views", "decoder", "scoring", "selector"},
        "algorithm",
    )
    _require(
        experiment,
        {"experiment_id", "run_id", "hypothesis", "baseline_reference", "cohort_reference", "seeds", "primary_metric", "compute_budget", "storage_budget", "requested_evidence"},
        "experiment",
    )
    _require(
        execution,
        {"challenge_path", "model_path", "native_config_dir", "artifact_root", "worker_count", "hard_deadline_seconds", "finalization_margin_seconds"},
        "execution",
    )
    if algorithm["solver_id"] != "fixed4plus4_d1":
        raise GovernanceError(f"unsupported experiment solver: {algorithm['solver_id']!r}")
    if not isinstance(algorithm["source_commit"], str) or len(algorithm["source_commit"]) != 40:
        raise GovernanceError("algorithm source_commit must be a full Git SHA")
    if not isinstance(algorithm["source_package_sha256"], str) or len(algorithm["source_package_sha256"]) != 64:
        raise GovernanceError("algorithm source_package_sha256 must be pinned")
    release_config = _mapping(algorithm["release_config"], "algorithm.release_config")
    _require(release_config, {"environment", "model_identity", "generation", "scoring", "runtime", "ttt24_recipe", "ttt48_recipe"}, "algorithm.release_config")
    views = _mapping(algorithm["views"], "algorithm.views")
    if set(views) != set(PORTFOLIO):
        raise GovernanceError("algorithm views must contain exactly TTT24 and TTT48")
    for source, expected_geometries in PORTFOLIO.items():
        rows = views[source]
        if not isinstance(rows, list) or len(rows) != 4:
            raise GovernanceError(f"{source} must contain four complete view specifications")
        expected = list(expected_geometries)
        actual = []
        for row in rows:
            item = _mapping(row, f"{source} view")
            if set(item) != {"geometry", "color_offset", "pair_order"}:
                raise GovernanceError("every view must explicitly specify geometry, color_offset, and pair_order")
            if item["color_offset"] != 0 or item["pair_order"] != "canonical":
                raise GovernanceError("fixed4plus4 D1 views must use color_offset=0 and canonical pair order")
            actual.append(item["geometry"])
        if actual != expected:
            raise GovernanceError(f"{source} view order differs from the frozen portfolio")
    if release_config["generation"].get("portfolio") != {key: list(value) for key, value in PORTFOLIO.items()}:
        raise GovernanceError("release_config portfolio differs from explicit view specifications")
    if algorithm["decoder"] != {"method": "greedy", "search_beams": 1}:
        raise GovernanceError("only the existing greedy fixed4plus4 solver is supported")
    if algorithm["selector"] != "D1_L_RRF_EXACT_B_RRF_TIE_BREAK":
        raise GovernanceError("selector differs from frozen D1")
    if int(execution["worker_count"]) != 4:
        raise GovernanceError("the existing shared solver requires four workers")
    if int(execution["hard_deadline_seconds"]) != int(release_config["runtime"]["hard_deadline_seconds"]):
        raise GovernanceError("execution deadline must match the explicit algorithm runtime contract")
    if int(execution["finalization_margin_seconds"]) != int(release_config["runtime"]["finalization_margin_seconds"]):
        raise GovernanceError("finalization margin must match the explicit algorithm runtime contract")
    resolved = copy.deepcopy(dict(config))
    resolved["schema_version"] = SCHEMA_VERSION
    resolved["config_sha256"] = digest(config)
    resolved["authoritative_modules"] = {
        "solver": "scripts.run_d1_release_4gpu.run_live",
        "finalizer": "scripts.build_d1_release_submission.finalize",
        "contract": "inference.d1_release_contract",
        "selector": "inference.selector_d1",
    }
    return resolved


def _clean_task(task: Any, task_id: str) -> dict[str, Any]:
    task = _mapping(task, f"task {task_id}")
    train = task.get("train")
    tests = task.get("test")
    if not isinstance(train, list) or not train or not isinstance(tests, list) or not tests:
        raise GovernanceError(f"task {task_id} must contain non-empty train and test lists")
    clean_train = []
    for index, example in enumerate(train):
        example = _mapping(example, f"task {task_id} train {index}")
        if "input" not in example or "output" not in example:
            raise GovernanceError(f"task {task_id} train {index} must contain input and output")
        clean_train.append({"input": copy.deepcopy(validate_grid(example["input"])), "output": copy.deepcopy(validate_grid(example["output"]))})
    clean_test = []
    for index, example in enumerate(tests):
        example = _mapping(example, f"task {task_id} test {index}")
        if "input" not in example:
            raise GovernanceError(f"task {task_id} test {index} lacks input")
        clean_test.append({"input": copy.deepcopy(validate_grid(example["input"]))})
    return {"train": clean_train, "test": clean_test}


def prepare_inference_bundle(
    challenges: Mapping[str, Any],
    cohort: Mapping[str, Any],
    bundle_dir: Path,
    *,
    source_dataset: Mapping[str, Any],
    solutions: Mapping[str, Any] | None = None,
    evaluation_output: Path | None = None,
) -> dict[str, Any]:
    """Create input-only test data and an optional physically separate target file."""
    if bundle_dir.exists():
        raise FileExistsError(f"refusing to overwrite inference bundle: {bundle_dir}")
    if solutions is not None:
        if evaluation_output is None:
            raise GovernanceError("solutions require a separate evaluation_output path")
        try:
            evaluation_output.resolve().relative_to(bundle_dir.resolve())
        except ValueError:
            pass
        else:
            raise GovernanceError("evaluation targets must be physically separate from the inference bundle")
    task_ids = cohort.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or len(task_ids) != len(set(task_ids)):
        raise GovernanceError("cohort task_ids must be a non-empty ordered unique list")
    missing = [task_id for task_id in task_ids if task_id not in challenges]
    if missing:
        raise GovernanceError(f"cohort references missing task IDs: {missing}")
    prepared: dict[str, Any] = {}
    tasks: dict[str, Any] = {}
    for task_id in task_ids:
        prepared[task_id] = _clean_task(challenges[task_id], task_id)
        tasks[task_id] = {
            "task_sha256": digest(challenges[task_id]),
            "inference_task_sha256": digest(prepared[task_id]),
            "test_index_structure": [
                {"test_index": index, "input_sha256": digest(example["input"])}
                for index, example in enumerate(prepared[task_id]["test"])
            ],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cohort_id": cohort.get("cohort_id"),
        "selection_method": cohort.get("selection_method"),
        "exposure_status": cohort.get("exposure_status"),
        "intended_use": cohort.get("intended_use"),
        "source_dataset": copy.deepcopy(dict(source_dataset)),
        "source_challenge_sha256": digest(challenges),
        "task_ids": list(task_ids),
        "task_ids_sha256": digest(task_ids),
        "tasks": tasks,
        "test_targets_in_bundle": False,
    }
    bundle_dir.mkdir(parents=True)
    # ARC task order is part of the experiment identity.  Do not let canonical
    # key sorting silently replace the frozen cohort order on disk.
    atomic_json(bundle_dir / "inference_tasks.json", prepared, sort_keys=False)
    atomic_json(bundle_dir / "cohort_manifest.json", manifest, sort_keys=False)
    if solutions is not None:
        assert evaluation_output is not None
        selected: dict[str, Any] = {}
        for task_id in task_ids:
            rows = solutions.get(task_id)
            if not isinstance(rows, list) or len(rows) != len(prepared[task_id]["test"]):
                raise GovernanceError(f"solution test-index structure mismatch for {task_id}")
            selected[task_id] = [copy.deepcopy(validate_grid(grid)) for grid in rows]
        if evaluation_output.exists():
            raise FileExistsError(f"refusing to overwrite evaluation targets: {evaluation_output}")
        atomic_json(
            evaluation_output,
            {"cohort_id": cohort.get("cohort_id"), "targets": selected, "source_solutions_sha256": digest(solutions)},
            sort_keys=False,
        )
    return manifest


def _set_manifest_state(run_dir: Path, state: str, **updates: Any) -> dict[str, Any]:
    if state not in RUN_STATES:
        raise GovernanceError(f"unknown run state: {state}")
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest["state"] = state
    atomic_json(path, manifest)
    return manifest


def prepare_run(config: Mapping[str, Any]) -> Path:
    assert_experiment_phase()
    resolved = resolve_experiment_config(config)
    execution = resolved["execution"]
    run_dir = Path(execution["artifact_root"]) / resolved["experiment"]["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"run identity already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "evaluation").mkdir()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": resolved["experiment"]["run_id"],
        "experiment_id": resolved["experiment"]["experiment_id"],
        "state": "PREPARED",
        "config_sha256": resolved["config_sha256"],
        "predictions_frozen": False,
        "solutions_opened": False,
        "exposure_status": resolved["experiment"].get("exposure_status", "UNKNOWN"),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "config_resolved.json", resolved)
    atomic_json(run_dir / "environment.json", {
        "source_commit": resolved["algorithm"]["source_commit"],
        "source_package_sha256": resolved["algorithm"]["source_package_sha256"],
        "environment": resolved["algorithm"]["release_config"]["environment"],
        "model_identity": resolved["algorithm"]["release_config"]["model_identity"],
        "evidence_class": "PREPARED_CPU_CONTRACT_ONLY",
    })
    append_event(run_dir / "events.jsonl", {"event": "RUN_PREPARED", "run_id": manifest["run_id"], "state": "PREPARED"})
    atomic_json(run_dir / "hashes.json", {"config_resolved.json": sha256_file(run_dir / "config_resolved.json")})
    return run_dir


def dry_run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate config and inference inputs without importing or calling the solver."""
    assert_experiment_phase()
    resolved = resolve_experiment_config(config)
    challenge_path = Path(resolved["execution"]["challenge_path"])
    challenges = json.loads(challenge_path.read_text(encoding="utf-8"))
    if not isinstance(challenges, Mapping) or not challenges:
        raise GovernanceError("workbench challenge must be a non-empty task object")
    for task_id, task in challenges.items():
        cleaned = _clean_task(task, task_id)
        if cleaned != task:
            raise GovernanceError("workbench challenge must already be an inference-only prepared bundle")
    return {
        "status": "DRY_RUN_PASS",
        "config_sha256": resolved["config_sha256"],
        "challenge_path": str(challenge_path),
        "task_count": len(challenges),
        "solver_called": False,
        "remote_action_started": False,
    }


Solver = Callable[[Path, dict[str, Any], Path, Path, Path], Mapping[str, Any]]
Finalizer = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]


def execute_run(run_dir: Path, solver: Callable[..., Mapping[str, Any]], finalizer: Finalizer) -> dict[str, Any]:
    """Invoke injected/shared solver and finalizer, then freeze non-submission artifacts."""
    assert_experiment_phase()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    prior_state = manifest.get("state")
    if prior_state not in {"PREPARED", "INCOMPLETE", "FAILED"}:
        raise GovernanceError(f"run is not resumable: {prior_state}")
    config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    hashes = json.loads((run_dir / "hashes.json").read_text(encoding="utf-8"))
    if config.get("config_sha256") != manifest.get("config_sha256"):
        raise GovernanceError("run config identity differs from the prepared manifest")
    if hashes.get("config_resolved.json") != sha256_file(run_dir / "config_resolved.json"):
        raise GovernanceError("run config_resolved.json changed after preparation")
    execution = config["execution"]
    challenge_path = Path(execution["challenge_path"])
    challenges = json.loads(challenge_path.read_text(encoding="utf-8"))
    if not isinstance(challenges, Mapping) or not challenges:
        raise GovernanceError("workbench challenge must be a non-empty task object")
    for task_id, task in challenges.items():
        cleaned = _clean_task(task, task_id)
        if cleaned != task:
            raise GovernanceError("workbench challenge must already be an inference-only prepared bundle")
    resume_count = int(manifest.get("resume_count", 0)) + int(prior_state != "PREPARED")
    _set_manifest_state(run_dir, "RUNNING", resume_count=resume_count)
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "RUN_STARTED" if prior_state == "PREPARED" else "RUN_RESUMED",
            "run_id": manifest["run_id"],
            "prior_state": prior_state,
            "resume_count": resume_count,
            "state": "RUNNING",
        },
    )
    try:
        release_config = config["algorithm"]["release_config"]
        artifact = dict(solver(
            challenge_path,
            release_config,
            Path(execution["model_path"]),
            Path(execution["native_config_dir"]),
            run_dir / "checkpoints",
            resume=True,
        ))
        selection, predictions, provenance = finalizer(challenges, release_config, artifact)
        atomic_json(run_dir / "candidates_frozen.json", artifact)
        atomic_json(run_dir / "scores_frozen.json", selection)
        atomic_json(run_dir / "predictions_frozen.json", predictions)
        hashes = {
            name: sha256_file(run_dir / name)
            for name in ("config_resolved.json", "candidates_frozen.json", "scores_frozen.json", "predictions_frozen.json")
        }
        atomic_json(run_dir / "hashes.json", hashes)
        _set_manifest_state(run_dir, "COMPLETE_FROZEN", predictions_frozen=True, solutions_opened=False, provenance=provenance)
        append_event(run_dir / "events.jsonl", {"event": "PREDICTIONS_FROZEN", "run_id": manifest["run_id"], "state": "COMPLETE_FROZEN", "predictions_sha256": hashes["predictions_frozen.json"]})
        return predictions
    except Exception as exc:
        _set_manifest_state(run_dir, "FAILED", error=f"{type(exc).__name__}: {exc}")
        append_event(run_dir / "events.jsonl", {"event": "RUN_FAILED", "run_id": manifest["run_id"], "error": f"{type(exc).__name__}: {exc}"})
        raise


def score_run(run_dir: Path, evaluation_targets: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "COMPLETE_FROZEN" or not manifest.get("predictions_frozen"):
        raise GovernanceError("predictions must be frozen before solutions are opened")
    report_path = run_dir / "evaluation" / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation: {report_path}")
    predictions = json.loads((run_dir / "predictions_frozen.json").read_text(encoding="utf-8"))
    target_artifact = json.loads(evaluation_targets.read_text(encoding="utf-8"))
    targets = target_artifact.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(predictions):
        raise GovernanceError("prediction/target task identity mismatch")
    top1 = top2 = outputs = 0
    task_top1 = task_top2 = 0
    for task_id, rows in predictions.items():
        truth = targets[task_id]
        if not isinstance(rows, list) or not isinstance(truth, list) or len(rows) != len(truth):
            raise GovernanceError(f"prediction/target test-index mismatch for {task_id}")
        hits1 = []; hits2 = []
        for row, target in zip(rows, truth, strict=True):
            validate_grid(target)
            hit1 = row["attempt_1"] == target
            hit2 = hit1 or row["attempt_2"] == target
            top1 += int(hit1); top2 += int(hit2); outputs += 1; hits1.append(hit1); hits2.append(hit2)
        task_top1 += int(all(hits1)); task_top2 += int(all(hits2))
    report = {"status": "COMPLETE_SCORED", "output_count": outputs, "top1_outputs": top1, "top2_outputs": top2, "task_count": len(predictions), "top1_tasks": task_top1, "top2_tasks": task_top2, "predictions_sha256": sha256_file(run_dir / "predictions_frozen.json"), "solutions_sha256": sha256_file(evaluation_targets), "exposure_status": manifest.get("exposure_status", "UNKNOWN")}
    atomic_json(report_path, report)
    _set_manifest_state(run_dir, "COMPLETE_SCORED", solutions_opened=True)
    append_event(run_dir / "events.jsonl", {"event": "RUN_SCORED", "run_id": manifest["run_id"], "state": "COMPLETE_SCORED"})
    return report


STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "INPUT_TOKENS": ("task_id", "task_sha256", "test_index_structure_sha256", "ordering_sha256", "view_sha256", "tokenizer_sha256", "prompt_template_sha256"),
    "ADAPTER": ("base_checkpoint_sha256", "initial_adapter_sha256", "training_inputs_sha256", "training_order_sha256", "recipe_sha256", "seed", "rng_convention", "source_commit", "environment_sha256"),
    "CANDIDATE": ("adapter_identity", "task_id", "task_sha256", "test_index", "view_sha256", "decoder_sha256", "seed", "task_seed_convention"),
    "LIKELIHOOD": ("adapter_identity", "prompt_tokens_sha256", "candidate_tokens_sha256", "eos_token_id", "score_definition", "environment_compatibility"),
    "SELECTION": ("candidate_artifact_sha256", "score_artifact_sha256", "selector_version", "selector_config_sha256"),
}


def compatibility_key(stage: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in STAGE_FIELDS:
        raise GovernanceError(f"unknown reuse stage: {stage}")
    missing = [field for field in STAGE_FIELDS[stage] if field not in provenance]
    if missing:
        return {"status": "UNKNOWN", "stage": stage, "missing": missing, "key": None}
    payload = {field: provenance[field] for field in STAGE_FIELDS[stage]}
    return {"status": "COMPLETE", "stage": stage, "missing": [], "key": digest(payload)}


def plan_reuse(stage: str, requested: Mapping[str, Any], available: Mapping[str, Any]) -> dict[str, Any]:
    if available.get("stage") != stage:
        return {"decision": "INCOMPATIBLE", "stage": stage, "reason": "artifact stage mismatch"}
    wanted = compatibility_key(stage, requested)
    existing = compatibility_key(stage, _mapping(available.get("provenance"), "available.provenance"))
    if wanted["status"] == "UNKNOWN" or existing["status"] == "UNKNOWN" or not available.get("artifact_sha256"):
        return {"decision": "UNKNOWN", "stage": stage, "requested": wanted, "available": existing, "reason": "missing provenance or artifact hash"}
    if wanted["key"] == existing["key"]:
        return {"decision": "REUSE_EXACT", "stage": stage, "compatibility_key": wanted["key"], "artifact_sha256": available["artifact_sha256"]}
    return {"decision": "COMPUTE_REQUIRED", "stage": stage, "requested_key": wanted["key"], "available_key": existing["key"]}


def index_artifact(path: Path, stage: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    key = compatibility_key(stage, provenance)
    return {"schema_version": SCHEMA_VERSION, "stage": stage, "path": str(path), "artifact_sha256": sha256_file(path), "provenance": copy.deepcopy(dict(provenance)), "compatibility": key}


def validate_inference_bundle(bundle_dir: Path) -> dict[str, Any]:
    forbidden = ("solution", "target", "answer", "credential", "kaggle.json", ".safetensors", ".bin", ".pt", ".pth")
    files = [path for path in bundle_dir.rglob("*") if path.is_file()]
    unsafe = [str(path.relative_to(bundle_dir)) for path in files if any(token in path.name.lower() for token in forbidden)]
    if unsafe:
        raise GovernanceError(f"inference bundle contains forbidden artifacts: {unsafe}")
    required = {"inference_tasks.json", "cohort_manifest.json"}
    names = {path.name for path in files}
    if not required.issubset(names):
        raise GovernanceError(f"inference bundle missing required files: {sorted(required - names)}")
    tasks = json.loads((bundle_dir / "inference_tasks.json").read_text(encoding="utf-8"))
    for task_id, task in tasks.items():
        if _clean_task(task, task_id) != task:
            raise GovernanceError("inference task bundle contains noncanonical or target-bearing test data")
    return {"status": "PASS", "file_count": len(files), "task_count": len(tasks)}
