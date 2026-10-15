from inference.rolling_resident_pool import RollingResidentQueue


def test_fifo_admission_different_termination_order_immediately_refills() -> None:
    queue = RollingResidentQueue([str(index) for index in range(11)], resident_capacity=8)
    assert queue.admit_available() == [str(index) for index in range(8)]
    queue.terminate("2"); assert queue.admit_available() == ["8"]
    queue.terminate("5"); assert queue.admit_available() == ["9"]
    queue.terminate("0"); assert queue.admit_available() == ["10"]
    assert queue.resident_ids == ("1", "3", "4", "6", "7", "8", "9", "10")
    assert queue.pending_ids == ()


def test_simultaneous_completion_refills_in_fifo_order_and_never_exceeds_cap() -> None:
    queue = RollingResidentQueue([str(index) for index in range(16)], resident_capacity=4)
    assert queue.admit_available() == ["0", "1", "2", "3"]
    for completed, replacement in [("3", "4"), ("1", "5"), ("0", "6"), ("2", "7")]:
        queue.terminate(completed)
        assert len(queue.resident_ids) == 3
        assert queue.admit_available() == [replacement]
        assert len(queue.resident_ids) == 4


def test_slow_resident_does_not_block_fifo_admission_after_other_cells_finish() -> None:
    queue = RollingResidentQueue([str(index) for index in range(8)], resident_capacity=2)
    assert queue.admit_available() == ["0", "1"]
    queue.terminate("1"); assert queue.admit_available() == ["2"]
    queue.terminate("2"); assert queue.admit_available() == ["3"]
    queue.terminate("3"); assert queue.admit_available() == ["4"]
    # Cell 0 remains resident throughout, but it never blocks replacement of
    # the other owner slot.
    assert queue.resident_ids == ("0", "4")


def test_runtime_callbacks_keep_full_cell_keys_and_not_augmentation_suffixes() -> None:
    """Mechanical regression for the real scheduler callback key contract."""
    from types import SimpleNamespace
    import inference.rolling_resident_pool as module

    original_reply = module._reply
    original_result = module.ready_result
    original_execute = module.execute_ready_forward
    seen: list[str] = []

    class Request:
        cache_key = ("same",)
        position = 7

    try:
        module._reply = lambda cell, _reply: setattr(cell, "request", None)
        module.ready_result = lambda _cell: SimpleNamespace(termination_reason="search_exhausted")
        module.execute_ready_forward = lambda **kwargs: ([None] * len(kwargs["selected"]), {})
        scheduler = module.run_rolling_resident_scheduler(
            model=object(),
            pending_ids=["task:o0:d24:aug16:a", "task:o0:d24:aug16:b", "task:o0:d24:aug16:c"],
            resident_capacity=2,
            physical_batch_ceiling=2,
            create_cell=lambda key: SimpleNamespace(cell_key=key, request=Request(), cache_owner=object(), state={}),
            consume_result=lambda key, _cell: seen.append(key),
            release_cell=lambda _key, _cell: None,
        )
    finally:
        module._reply = original_reply
        module.ready_result = original_result
        module.execute_ready_forward = original_execute
    assert seen == ["task:o0:d24:aug16:a", "task:o0:d24:aug16:b", "task:o0:d24:aug16:c"]
    assert [row["cell_key"] for row in scheduler["events"] if row["event"] == "ADMIT"] == seen


def test_active_budget_charges_full_forward_to_each_participant_but_not_queue_time(monkeypatch):
    from types import SimpleNamespace
    import inference.rolling_resident_pool as module

    clock = [0.0]
    cells = []
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])

    def create(key):
        clock[0] += 1000.0  # Simulated admission/queue delay is not cell work.
        cell = SimpleNamespace(cell_key=key,
            request=SimpleNamespace(cache_key=('same',), position=7), cache_owner=object(),
            state={'active_time_accounting': True, 'active_elapsed_seconds': 0.0},
            active_elapsed_seconds=0.0)
        cells.append(cell)
        return cell

    def forward(**kwargs):
        clock[0] += 2.0
        return [None] * len(kwargs['selected']), {}

    monkeypatch.setattr(module, 'execute_ready_forward', forward)
    monkeypatch.setattr(module, '_reply', lambda cell, _: setattr(cell, 'request', None))
    monkeypatch.setattr(module, 'ready_result', lambda _: SimpleNamespace(termination_reason='search_exhausted'))
    module.run_rolling_resident_scheduler(model=object(), pending_ids=['a', 'b', 'c'],
        resident_capacity=2, physical_batch_ceiling=2, create_cell=create,
        consume_result=lambda *_: None, release_cell=lambda *_: None,
        synchronize_forward=lambda: None)
    assert [c.state['active_elapsed_seconds'] for c in cells] == [2.0, 2.0, 2.0]
    assert [c.active_elapsed_seconds for c in cells] == [2.0, 2.0, 2.0]
