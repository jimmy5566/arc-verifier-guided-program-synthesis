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
from inference.root_adaptive_batch_policy import FairCompatibilitySelector, RootAwareAdmissionQueue


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
    admission_policy: str = "fifo",
    root_lengths: dict[str, int] | None = None,
    safe_batch_ceiling: Callable[[int], int] | None = None,
    fairness_max_wait: int = 3,
    cache_pack_observer: Callable[[str, dict[str, Any]], None] | None = None,
    release_batch_temporaries_for_audit: bool = False,
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
    if admission_policy not in {"fifo", "root_aware"}:
        raise ValueError("admission_policy must be fifo or root_aware")
    pending = tuple(str(item) for item in pending_ids)
    if admission_policy == "root_aware":
        if root_lengths is None:
            raise ValueError("root-aware admission requires root_lengths")
        queue: Any = RootAwareAdmissionQueue(
            pending, root_lengths={str(key): int(value) for key, value in root_lengths.items()},
            resident_capacity=resident_capacity,
        )
    else:
        queue = RollingResidentQueue(pending, resident_capacity)

    residents: OrderedDict[str, ReadyCell] = OrderedDict()
    events: list[dict[str, Any]] = []
    histogram: dict[int, int] = {}
    physical_forwards = 0
    logical_advances = 0
    event_index = 0
    max_resident_count = 0
    forward_timing = {"model_call_seconds": 0.0, "cache_pack_seconds": 0.0, "cache_adoption_seconds": 0.0, "elapsed_seconds": 0.0}
    selector = FairCompatibilitySelector(max_wait=fairness_max_wait) if safe_batch_ceiling is not None else None
    root_by_key = {str(key): int(value) for key, value in (root_lengths or {}).items()}

    def emit(event: str, *, cell_key: str | None, reason: str | None = None,
             physical_forward_index: int | None = None, physical_batch: int | None = None,
             selected_keys: list[str] | None = None, compatibility_class: str | None = None,
             compatibility_wait_before: int | None = None, safe_ceiling: int | None = None,
             scheduling_cycle_id: int | None = None, forward_index_within_cycle: int | None = None,
             compatible_class_sizes: tuple[tuple[str, int], ...] | None = None,
             selected_class_size: int | None = None, request_position: int | None = None) -> None:
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
            "admission_policy": admission_policy,
            "active_root_length": getattr(queue, "active_root_length", None),
            "resident_root_lengths": [root_by_key[key] for key in residents if key in root_by_key],
            "selected_root_lengths": [root_by_key[key] for key in (selected_keys or []) if key in root_by_key],
            "compatibility_class": compatibility_class,
            "compatibility_wait_before": compatibility_wait_before,
            "safe_batch_ceiling": safe_ceiling,
            # This is recomputed from the live READY state immediately before
            # every physical forward; it is never a sticky profile width.
            "scheduling_cycle_id": scheduling_cycle_id,
            "forward_index_within_cycle": forward_index_within_cycle,
            "compatible_class_sizes": [
                {"compatibility_class": class_key, "size": size}
                for class_key, size in (compatible_class_sizes or ())
            ],
            "selected_class_size": selected_class_size,
            "actual_batch_width": physical_batch,
            "request_position": request_position,
            "current_cache_length": request_position,
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
        decision = selector.select(residents.values(), safe_batch_ceiling) if selector is not None else None
        selected = list(decision.selected) if decision is not None else _compatible_group_in_frozen_order(
            list(residents.values()), int(physical_batch_ceiling),
        )
        if not selected:
            raise RuntimeError("rolling resident pool has owners but no READY request")
        requests = [cell.request for cell in selected]
        if any(request is None for request in requests):
            raise RuntimeError("selected rolling cell has no request")
        physical_forwards += 1
        physical_batch = len(selected)
        emit("FORWARD", cell_key=None, physical_batch=physical_batch,
             selected_keys=[cell.cell_key for cell in selected],
             compatibility_class=decision.class_key if decision is not None else None,
             compatibility_wait_before=decision.class_wait_before if decision is not None else None,
             safe_ceiling=decision.safe_ceiling if decision is not None else physical_batch_ceiling,
             scheduling_cycle_id=decision.scheduling_cycle_id if decision is not None else physical_forwards - 1,
             forward_index_within_cycle=decision.forward_index_within_cycle if decision is not None else 0,
             compatible_class_sizes=decision.compatible_class_sizes if decision is not None else None,
             selected_class_size=decision.selected_class_size if decision is not None else physical_batch,
             request_position=int(requests[0].position) if requests else None)
        replies, telemetry = execute_ready_forward(
            model=model,
            selected=selected,
            requests=requests,  # type: ignore[arg-type]
            streaming_split_and_adopt=True,
            cache_pack_observer=cache_pack_observer,
            release_batch_temporaries_for_audit=release_batch_temporaries_for_audit,
        )
        for key in forward_timing:
            forward_timing[key] += float(telemetry.get(key, 0.0))
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
        "admission_policy": admission_policy,
        "compatibility_wait_counts": selector.wait_counts() if selector is not None else {},
        "forward_timing": forward_timing,
        "events": events,
        "status": "COMPLETE",
    }
