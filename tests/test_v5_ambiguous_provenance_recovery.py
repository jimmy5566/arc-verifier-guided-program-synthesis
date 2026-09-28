"""CPU contracts for deterministic V5 provenance recovery."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "v5_recovery", Path(__file__).parents[1] / "scripts" / "recover_v5_ambiguous_provenance.py"
)
assert SPEC and SPEC.loader
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)

KEY = ("deadbeef", 0, 24, "identity")


def db(**overrides):
    value = {
        "task_id": KEY[0], "output_index": KEY[1], "depth": KEY[2], "view": KEY[3], "status": "DONE",
        "checkpoint_sha256": "a" * 64, "config_sha256": RECOVERY.CONFIG_SHA,
        "candidate_count": 2, "nodes_expanded": 10, "model_forwards": 11, "tokens_advanced": 12,
        "frontier_floor_activation_count": 3, "runtime_seconds": 4.0,
    }
    value.update(overrides)
    return value


def record(path="/tmp/a.json", **overrides):
    raw = {
        "task_id": KEY[0], "output_index": KEY[1], "depth": KEY[2], "view": KEY[3],
        "checkpoint_sha256": "a" * 64, "decoder_config_sha256": RECOVERY.CONFIG_SHA,
        "candidate_count": 2, "nodes_expanded": 10, "model_forwards": 11, "tokens_advanced": 12,
        "frontier_floor_activation_count": 3, "runtime_seconds": 4.0, "execution_engine": "INDEPENDENT_FAST_2_PER_GPU",
        "candidates": [{"candidate_id": 0, "cumulative_nll": 1.0, "canonical_candidate": [[1]]}],
    }
    raw.update(overrides)
    item = RECOVERY.normalized_record(raw, Path(path), 0)
    assert item is not None
    item["artifact_sha256"] = "b" * 64
    return item


def resolve(rows, row=None):
    history, recovered, unresolved, resolutions, states = RECOVERY.resolve({KEY: row or db()}, rows, {KEY: "AMBIGUOUS"})
    return history, recovered, unresolved, resolutions, states


def test_unique_exact_artifact_promotes():
    _, recovered, unresolved, _, _ = resolve([record()])
    assert recovered[0]["match_class"] == "EXACT_LINK" and not unresolved


def test_duplicate_identical_artifacts_promote():
    left, right = record("/tmp/a.json"), record("/tmp/b.json")
    _, recovered, unresolved, _, _ = resolve([left, right])
    assert recovered[0]["match_class"] == "DUPLICATE_IDENTICAL_LINK" and not unresolved


def test_duplicate_conflicting_artifacts_stay_unresolved():
    left, right = record(), record("/tmp/b.json", candidates=[{"candidate_id": 9, "cumulative_nll": 2.0, "canonical_candidate": [[9]]}])
    _, recovered, unresolved, _, _ = resolve([left, right])
    assert not recovered and unresolved[0]["reason"] == "MULTIPLE_CONFLICTING_ARTIFACTS"


def test_orphan_artifact_without_db_pointer_is_eligible():
    _, recovered, unresolved, _, _ = resolve([record("/tmp/orphan.json")])
    assert recovered and not unresolved


def test_checkpoint_mismatch_stays_unresolved():
    _, recovered, unresolved, _, _ = resolve([record(checkpoint_sha256="c" * 64)])
    assert not recovered and unresolved[0]["reason"] == "CHECKPOINT_CONFLICT"


def test_candidate_count_mismatch_is_telemetry_conflict():
    _, recovered, unresolved, _, _ = resolve([record(candidate_count=7)])
    assert not recovered and unresolved[0]["reason"] == "TELEMETRY_CONFLICT"


def test_oom_then_valid_heavy_retry_selects_valid_record():
    oom = record("/tmp/oom.json", candidate_count=0, termination_reason="oom")
    good = record("/tmp/heavy.json", execution_engine="INDEPENDENT_SINGLE_PER_GPU_REPAIR")
    _, recovered, unresolved, _, _ = resolve([oom, good])
    assert recovered[0]["selected_artifact_path"].replace("\\", "/") == "/tmp/heavy.json" and not unresolved


def test_multiple_completed_conflicting_retries_are_not_forced():
    first, second = record("/tmp/first.json"), record("/tmp/second.json", candidates=[{"candidate_id": 8, "cumulative_nll": 2.0, "canonical_candidate": [[8]]}])
    _, recovered, unresolved, _, _ = resolve([first, second])
    assert not recovered and unresolved


def test_no_artifact_is_explicit():
    _, recovered, unresolved, _, _ = resolve([])
    assert not recovered and unresolved[0]["reason"] == "NO_ARTIFACT"


def test_gold_independence_by_contract():
    # Resolver signature deliberately receives no Gold object or path.
    first = resolve([record()])[1]
    second = resolve([record()])[1]
    assert first == second
