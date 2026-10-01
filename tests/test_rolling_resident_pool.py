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
