from scripts.phase2_backend_contract import BACKENDS, execution_for_backend, valid_frozen_artifact


def test_backend_contract_is_frozen_and_explicit() -> None:
    assert BACKENDS == ("reference", "compaction", "static_kv", "torch_compile", "cuda_graph", "best_combined")
    assert execution_for_backend("reference", successful=set()) == ("baseline", None)
    assert execution_for_backend("compaction", successful=set()) == ("active_compaction", None)
    assert execution_for_backend("cuda_graph", successful=set())[0] is None
    assert execution_for_backend("best_combined", successful={"static_kv", "torch_compile"}) == ("static_kv_torch_compile", None)


def test_frozen_candidate_artifact_contract() -> None:
    task_id = "5dbc8537"
    assert valid_frozen_artifact({"status": "EVAL3_RUNTIME_OPT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING", "task_ids": [task_id], "records": {task_id: {}}}, task_id)
    assert not valid_frozen_artifact({"status": "wrong", "task_ids": [task_id], "records": {task_id: {}}}, task_id)
