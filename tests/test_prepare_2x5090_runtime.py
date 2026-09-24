from scripts.prepare_2x5090_runtime import worker_plan


def test_two_blackwell_workers_are_independent_and_pinned() -> None:
    plan = worker_plan(
        [
            {"name": "NVIDIA GeForce RTX 5090", "capability": [12, 0]},
            {"name": "NVIDIA GeForce RTX 5090", "capability": [12, 0]},
        ]
    )
    assert plan["worker_count"] == 2
    assert [worker["cuda_visible_devices"] for worker in plan["workers"]] == ["0", "1"]
    assert all(worker["model_instances"] == 1 for worker in plan["workers"])
    assert not any(worker["tensor_parallelism"] for worker in plan["workers"])


def test_wrong_inventory_is_rejected() -> None:
    try:
        worker_plan([{"name": "NVIDIA GeForce RTX 5090", "capability": [12, 0]}])
    except ValueError as error:
        assert "GPU_INVENTORY_MISMATCH" in str(error)
    else:
        raise AssertionError("expected an inventory failure")
