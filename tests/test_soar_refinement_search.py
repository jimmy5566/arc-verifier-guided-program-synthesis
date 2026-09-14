import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("soar_search", ROOT / "scripts" / "run_soar_refinement_search.py")
search = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(search)


def test_atomic_state_roundtrip_and_summary_is_target_blind(tmp_path: Path):
    output = tmp_path / "out"; task_path = output / "tasks" / "task.json"
    search.atomic(task_path, {"task_id": "task", "candidate_programs": [{"candidate_id": "r0-c0", "all_train_exact": True, "verification": {"train_pass_count": 2}}]})
    state = json.loads(task_path.read_text(encoding="utf-8"))
    assert state["candidate_programs"][0]["candidate_id"] == "r0-c0"
    source = (ROOT / "scripts" / "run_soar_refinement_search.py").read_text(encoding="utf-8")
    assert "load_solutions" not in source
    assert "test_target_policy" in source


def test_search_uses_official_budget_and_sequential_model_sandbox_lifecycle():
    source = (ROOT / "scripts" / "run_soar_refinement_search.py").read_text(encoding="utf-8")
    assert "initial_max_new_tokens\": 4096" in source
    assert "refinement_max_new_tokens\": 2048" in source
    main_body = source[source.index("def main()") :]
    assert main_body.index("_launch(initial_jobs") < main_body.index("_verify_unverified(tasks")


def test_task_checkpoint_has_exactly_one_generation_writer():
    jobs = [
        {"task_id": "a", "candidate_id": "r0-c0"},
        {"task_id": "a", "candidate_id": "r0-c1"},
        {"task_id": "b", "candidate_id": "r0-c0"},
        {"task_id": "b", "candidate_id": "r0-c1"},
    ]
    shards = search._shard_by_task(jobs)
    locations = {job["candidate_id"] + job["task_id"]: index for index, shard in enumerate(shards) for job in shard}
    assert locations["r0-c0a"] == locations["r0-c1a"]
    assert locations["r0-c0b"] == locations["r0-c1b"]


def test_reverification_is_train_only():
    source = (ROOT / "scripts" / "reverify_soar_search_artifact.py").read_text(encoding="utf-8")
    assert "load_solutions" not in source
    assert "challenge-path" in source
    assert "safe_allowlisted_helpers_stdlib_v2" in source
