"""FIFO rolling-resident execution for non-S AUG16 ReadyCell surfaces.

This module changes *only* owner admission and READY selection.  Each actual
incremental forward is delegated to :func:`execute_ready_forward` from the
validated Clean-HF dynamic-ready engine; it does not implement a second
decoder, cache packer, or DFS coroutine path.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from inference.nvarc_turbodfs_dynamic_ready import (
    ReadyCell,
    _compatible_group_in_frozen_order,
    _reply,
    execute_ready_forward,
    ready_result,
)


@dataclass(frozen=True)
class PoolEvent:
    """Pure, serializable resident-pool state transition."""

    event_index: int
    physical_forward_index: int
    event: str
    cell_key: str | None
    resident_count: int
    pending_count: int
    termination_reason: str | None = None


class RollingResidentQueue:
    """Deterministic FIFO admission bookkeeping independent of CUDA/DFS.

    The queue is intentionally small and separately testable.  It never sees
    caches; callers may create an owner only for IDs returned by ``admit``.
    """

    def __init__(self, pending_ids: Iterable[str], resident_capacity: int) -> None:
        ids = tuple(str(item) for item in pending_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("rolling resident queue requires distinct pending IDs")
        if int(resident_capacity) <= 0 or int(resident_capacity) > len(ids):
            raise ValueError("resident capacity must be positive and no larger than the frozen queue")
        self._pending: deque[str] = deque(ids)
        self._resident: OrderedDict[str, None] = OrderedDict()
        self._capacity = int(resident_capacity)

    @property
    def resident_ids(self) -> tuple[str, ...]:
        return tuple(self._resident)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def capacity(self) -> int:
        return self._capacity

    def admit_available(self) -> list[str]:
        admitted: list[str] = []
        while self._pending and len(self._resident) < self._capacity:
            cell_key = self._pending.popleft()
            self._resident[cell_key] = None
            admitted.append(cell_key)
        return admitted

    def terminate(self, cell_key: str) -> None:
        if cell_key not in self._resident:
            raise RuntimeError(f"attempted to terminate non-resident cell: {cell_key}")
        del self._resident[cell_key]

    def complete(self) -> bool:
        return not self._pending and not self._resident


def run_rolling_resident_scheduler(
    *,
    model: Any,
    pending_ids: Iterable[str],
    resident_capacity: int,
    physical_batch_ceiling: int,
    create_cell: Callable[[str], ReadyCell],
    consume_result: Callable[[str, ReadyCell], None],
    release_cell: Callable[[str, ReadyCell], None],
    memory_snapshot: Callable[[], dict[str, Any]] | None = None,
    cache_summary: Callable[[Iterable[ReadyCell]], dict[str, Any]] | None = None,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run a FIFO rolling pool through the common physical-forward path.

    A newly-admitted cell starts with a scalar root prefill and becomes a
    resident owner.  At most ``resident_capacity`` owners exist.  Every
    completed logical cell is consumed, released, and replaced immediately in
    the frozen FIFO order, even when multiple cells finish after one physical
    forward.  Selection does not wait for a full batch: it selects the largest
    compatible subset of *currently READY resident cells* up to the ceiling.
    """
    if int(physical_batch_ceiling) <= 0 or int(physical_batch_ceiling) > int(resident_capacity):
        raise ValueError("physical batch ceiling must be in [1, resident_capacity]")

    queue = RollingResidentQueue(pending_ids, resident_capacity)
    residents: OrderedDict[str, ReadyCell] = OrderedDict()
    events: list[dict[str, Any]] = []
    histogram: dict[int, int] = {}
    physical_forwards = 0
    logical_advances = 0
    event_index = 0
    max_resident_count = 0

    def emit(event: str, *, cell_key: str | None, reason: str | None = None,
             physical_forward_index: int | None = None, physical_batch: int | None = None,
             selected_keys: list[str] | None = None) -> None:
        nonlocal event_index, max_resident_count
        event_index += 1
        max_resident_count = max(max_resident_count, len(residents))
        payload: dict[str, Any] = {
            "event_index": event_index,
            "physical_forward_index": physical_forwards if physical_forward_index is None else physical_forward_index,
            "event": event,
            "cell_key": cell_key,
            "resident_count": len(residents),
            "pending_count": len(queue.pending_ids),
            "resident_capacity": queue.capacity,
            "termination_reason": reason,
            "physical_batch": physical_batch,
            "selected_cell_keys": selected_keys or [],
        }
        if memory_snapshot is not None:
            payload.update({f"memory_{key}": value for key, value in memory_snapshot().items()})
        if cache_summary is not None:
            payload.update({f"cache_{key}": value for key, value in cache_summary(residents.values()).items()})
        events.append(payload)
        if event_sink is not None:
            event_sink(payload)

    def admit_available(*, physical_index: int) -> None:
        for cell_key in queue.admit_available():
            # The owner is created only after this ID has been removed from the
            # pending queue and admitted.  Pending IDs never own a GPU cache.
            cell = create_cell(cell_key)
            if cell.request is None or cell.cache_owner is None:
                raise RuntimeError(f"admitted cell did not produce a READY request: {cell_key}")
            residents[cell_key] = cell
            emit("ADMIT", cell_key=cell_key, physical_forward_index=physical_index)

    admit_available(physical_index=0)
    while residents:
        selected = _compatible_group_in_frozen_order(list(residents.values()), int(physical_batch_ceiling))
        if not selected:
            raise RuntimeError("rolling resident pool has owners but no READY request")
        requests = [cell.request for cell in selected]
        if any(request is None for request in requests):
            raise RuntimeError("selected rolling cell has no request")
        physical_forwards += 1
        physical_batch = len(selected)
        emit("FORWARD", cell_key=None, physical_batch=physical_batch,
             selected_keys=[cell.cell_key for cell in selected])
        replies, _telemetry = execute_ready_forward(
            model=model,
            selected=selected,
            requests=requests,  # type: ignore[arg-type]
            streaming_split_and_adopt=True,
        )
        histogram[physical_batch] = histogram.get(physical_batch, 0) + 1
        logical_advances += physical_batch
        for cell, reply in zip(selected, replies, strict=True):
            _reply(cell, reply)
        for cell in selected:
            if cell.request is not None:
                continue
            result = ready_result(cell)
            consume_result(cell.cell_key, cell)
            emit("TERMINATE", cell_key=cell.cell_key, reason=result.termination_reason,
                 physical_batch=physical_batch, selected_keys=[item.cell_key for item in selected])
            queue.terminate(cell.cell_key)
            resident = residents.pop(cell.cell_key)
            release_cell(cell.cell_key, resident)
            emit("RELEASE", cell_key=cell.cell_key, reason=result.termination_reason,
                 physical_batch=physical_batch, selected_keys=[item.cell_key for item in selected])
            admit_available(physical_index=physical_forwards)

    if not queue.complete():  # pragma: no cover - defensive invariant
        raise RuntimeError("rolling resident scheduler ended before FIFO queue drained")
    return {
        "physical_forwards": physical_forwards,
        "logical_advances": logical_advances,
        "physical_batch_histogram": {str(width): count for width, count in sorted(histogram.items())},
        "mean_effective_batch": logical_advances / physical_forwards if physical_forwards else 0.0,
        "resident_capacity": queue.capacity,
        "max_resident_count": max_resident_count,
        "events": events,
        "status": "COMPLETE",
    }
