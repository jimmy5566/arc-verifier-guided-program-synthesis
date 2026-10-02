"""Deterministic, target-blind root/cache-length batching policy primitives.

This module contains no CUDA queries and no decoder outcome signals.  It owns
only three pieces of scheduler bookkeeping: a frozen length lookup table,
root-length-aware admission order, and bounded-wait selection among exact
READY compatibility classes.  The physical model forward remains in
``nvarc_turbodfs_dynamic_ready.execute_ready_forward``.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class RootBatchPolicyEntry:
    """One inclusive cache-length boundary from a measured calibration."""

    max_cache_length: int
    resident_capacity: int
    physical_batch_ceiling: int

    def __post_init__(self) -> None:
        if int(self.max_cache_length) <= 0:
            raise ValueError("max_cache_length must be positive")
        if int(self.resident_capacity) <= 0:
            raise ValueError("resident_capacity must be positive")
        if not 1 <= int(self.physical_batch_ceiling) <= int(self.resident_capacity):
            raise ValueError("physical_batch_ceiling must be in [1, resident_capacity]")


class RootBatchPolicy:
    """Validated immutable cache-length lookup table.

    The table is deliberately monotone: moving to a longer cache bucket may
    not increase either resident capacity or physical batch ceiling.  It is a
    deterministic engineering policy, never a free-VRAM runtime heuristic.
    """

    def __init__(self, entries: Sequence[RootBatchPolicyEntry]) -> None:
        if not entries:
            raise ValueError("root batch policy requires at least one entry")
        self._entries = tuple(entries)
        previous: RootBatchPolicyEntry | None = None
        for entry in self._entries:
            if previous is not None:
                if entry.max_cache_length <= previous.max_cache_length:
                    raise ValueError("policy cache boundaries must strictly increase")
                if entry.resident_capacity > previous.resident_capacity:
                    raise ValueError("resident capacity may not increase with cache length")
                if entry.physical_batch_ceiling > previous.physical_batch_ceiling:
                    raise ValueError("physical ceiling may not increase with cache length")
            previous = entry

    @property
    def entries(self) -> tuple[RootBatchPolicyEntry, ...]:
        return self._entries

    def lookup(self, cache_length: int) -> RootBatchPolicyEntry:
        value = int(cache_length)
        if value <= 0:
            raise ValueError("cache length must be positive")
        for entry in self._entries:
            if value <= entry.max_cache_length:
                return entry
        raise RuntimeError(f"cache length {value} exceeds frozen calibrated policy")

    def to_json(self) -> list[dict[str, int]]:
        return [asdict(entry) for entry in self._entries]


class RootAwareAdmissionQueue:
    """Admission queue which clusters same-root AUG16 owners without dropping order.

    Buckets are frozen before GPU work.  The first bucket is the largest root
    class (ties: earliest original AUG16 index).  Refills favour the active
    bucket while it still has pending members, then deterministically move to
    the next largest remaining bucket.  Each bucket itself retains frozen
    AUG16 order.
    """

    def __init__(self, pending_ids: Iterable[str], *, root_lengths: dict[str, int], resident_capacity: int) -> None:
        ids = tuple(str(item) for item in pending_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("root-aware admission requires distinct nonempty IDs")
        if not 1 <= int(resident_capacity) <= len(ids):
            raise ValueError("resident capacity must be within pending queue size")
        missing = [key for key in ids if key not in root_lengths]
        if missing:
            raise ValueError(f"root lengths missing for {missing[:3]}")
        self._capacity = int(resident_capacity)
        self._root_lengths = {key: int(root_lengths[key]) for key in ids}
        if any(value <= 0 for value in self._root_lengths.values()):
            raise ValueError("root lengths must be positive")
        self._order = {key: index for index, key in enumerate(ids)}
        buckets: dict[int, deque[str]] = defaultdict(deque)
        for key in ids:
            buckets[self._root_lengths[key]].append(key)
        self._pending_by_root = dict(buckets)
        self._resident: OrderedDict[str, None] = OrderedDict()
        self._active_root: int | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def resident_ids(self) -> tuple[str, ...]:
        return tuple(self._resident)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(key for _root, keys in self._pending_by_root.items() for key in keys)

    @property
    def active_root_length(self) -> int | None:
        return self._active_root

    def root_length(self, cell_key: str) -> int:
        return self._root_lengths[str(cell_key)]

    def _next_root(self) -> int | None:
        nonempty = [(root, keys) for root, keys in self._pending_by_root.items() if keys]
        if not nonempty:
            return None
        if self._active_root is not None and self._pending_by_root.get(self._active_root):
            return self._active_root
        # Largest remaining bucket, then earliest immutable AUG16 position.
        root, _keys = min(
            nonempty,
            key=lambda item: (-len(item[1]), min(self._order[key] for key in item[1])),
        )
        self._active_root = root
        return root

    def admit_available(self) -> list[str]:
        admitted: list[str] = []
        while len(self._resident) < self._capacity:
            root = self._next_root()
            if root is None:
                break
            key = self._pending_by_root[root].popleft()
            self._resident[key] = None
            admitted.append(key)
        return admitted

    def terminate(self, cell_key: str) -> None:
        key = str(cell_key)
        if key not in self._resident:
            raise RuntimeError(f"attempted to terminate non-resident cell: {key}")
        del self._resident[key]

    def complete(self) -> bool:
        return not self._resident and not any(self._pending_by_root.values())


@dataclass(frozen=True)
class CompatibilityDecision:
    selected: tuple[Any, ...]
    class_key: str
    class_wait_before: int
    class_wait_after: int
    eligible_class_count: int
    selected_width: int
    safe_ceiling: int


class FairCompatibilitySelector:
    """Largest-compatible scheduling with deterministic bounded waiting.

    A class with a persistent READY request is served after at most
    ``max_wait`` other physical forwards.  Until that bound is reached, the
    selector maximises the safely executable compatible width.  This prevents
    a small 7-way class from being permanently starved by a busy 9-way class
    while retaining the throughput preference for the latter.
    """

    def __init__(self, *, max_wait: int = 3) -> None:
        if int(max_wait) < 1:
            raise ValueError("max_wait must be at least one")
        self._max_wait = int(max_wait)
        self._waits: dict[str, int] = {}

    @staticmethod
    def _class_key(cell: Any) -> tuple[str, int]:
        request = cell.request
        if request is None:
            raise ValueError("compatibility selector received an unready cell")
        return (repr(request.cache_key), int(request.position))

    @staticmethod
    def _key_text(key: tuple[str, int]) -> str:
        return f"cache={key[0]}|position={key[1]}"

    def select(self, cells: Iterable[Any], safe_ceiling: Callable[[int], int]) -> CompatibilityDecision | None:
        grouped: OrderedDict[tuple[str, int], list[Any]] = OrderedDict()
        for cell in cells:
            if cell.request is None:
                continue
            key = self._class_key(cell)
            grouped.setdefault(key, []).append(cell)
        if not grouped:
            self._waits.clear()
            return None
        active_text = {self._key_text(key) for key in grouped}
        self._waits = {key: value for key, value in self._waits.items() if key in active_text}
        candidates: list[tuple[tuple[str, int], list[Any], int, int, int]] = []
        for key, group in grouped.items():
            ceiling = int(safe_ceiling(key[1]))
            if ceiling <= 0:
                raise RuntimeError("frozen batch policy returned a non-positive ceiling")
            text = self._key_text(key)
            wait = self._waits.get(text, 0)
            width = min(len(group), ceiling)
            candidates.append((key, group, ceiling, wait, width))
        overdue = [item for item in candidates if item[3] >= self._max_wait]
        if overdue:
            # Oldest eligible class wins; width and frozen compatibility key
            # resolve ties deterministically.
            selected = min(overdue, key=lambda item: (-item[3], -item[4], self._key_text(item[0])))
        else:
            # Throughput first until any class reaches the bounded-wait limit.
            selected = min(candidates, key=lambda item: (-item[4], self._key_text(item[0])))
        selected_key, group, ceiling, wait_before, width = selected
        selected_text = self._key_text(selected_key)
        for key, _group, _ceiling, current_wait, _width in candidates:
            text = self._key_text(key)
            self._waits[text] = 0 if text == selected_text else current_wait + 1
        return CompatibilityDecision(
            selected=tuple(group[:width]), class_key=selected_text,
            class_wait_before=wait_before, class_wait_after=self._waits[selected_text],
            eligible_class_count=len(candidates), selected_width=width, safe_ceiling=ceiling,
        )

    def wait_counts(self) -> dict[str, int]:
        return dict(sorted(self._waits.items()))
