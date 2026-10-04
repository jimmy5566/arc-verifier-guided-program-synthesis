"""Deterministic logical work ordering for retained DFS branches.

The policies in this module deliberately know nothing about model calls,
retention, cache ownership, prompts, or physical batching.  A decoder first
decides which successors are retained under the frozen rule, then hands their
logical work items to one of these policies.  Changing policy therefore cannot
add or remove a retained successor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol


SearchOrderName = Literal["CURRENT_DFS", "FAIR_DFS_Q64", "REGRET_BAND_FAIR_Q64"]


@dataclass(frozen=True)
class RetainedWorkItem:
    """A cache-free continuation frame selected after frozen retention."""

    work_item_id: int
    root_branch_id: int
    parent_work_item_id: int | None
    cumulative_nll: float
    cumulative_regret: float
    insertion_order: int

    @property
    def regret_band(self) -> int:
        if self.cumulative_regret < 1.0:
            return 0
        if self.cumulative_regret < 2.0:
            return 1
        if self.cumulative_regret < 3.0:
            return 2
        if self.cumulative_regret <= 4.0:
            return 3
        raise ValueError("a non-retained work item has no regret band")


class SearchOrderPolicy(Protocol):
    """Policy boundary used by the shared logical search core."""

    name: SearchOrderName

    def push(self, item: RetainedWorkItem) -> None: ...

    def pop(self) -> RetainedWorkItem | None: ...

    def note_expansion(self, item: RetainedWorkItem) -> None: ...

    def pending(self) -> tuple[RetainedWorkItem, ...]: ...


def _key(item: RetainedWorkItem) -> tuple[float, int, int]:
    return (float(item.cumulative_nll), int(item.insertion_order), int(item.work_item_id))


class CurrentDFS:
    """Historical cumulative-NLL DFS ordering represented as an explicit stack."""

    name: SearchOrderName = "CURRENT_DFS"

    def __init__(self) -> None:
        self._stack: list[RetainedWorkItem] = []

    def push(self, item: RetainedWorkItem) -> None:
        # The legacy decoder sorts siblings low-NLL first, then recurses into
        # the first sibling.  Reversing the sorted insertion makes a simple
        # LIFO stack reproduce that depth-first selection.
        index = 0
        while index < len(self._stack) and _key(self._stack[index]) >= _key(item):
            index += 1
        self._stack.insert(index, item)

    def pop(self) -> RetainedWorkItem | None:
        return self._stack.pop() if self._stack else None

    def note_expansion(self, item: RetainedWorkItem) -> None:
        del item

    def pending(self) -> tuple[RetainedWorkItem, ...]:
        return tuple(reversed(self._stack))


class _FairQueues:
    """Round-robin root-subtree queues with a fixed expanded-node quantum."""

    def __init__(self, quantum: int) -> None:
        if quantum <= 0:
            raise ValueError("fair DFS quantum must be positive")
        self.quantum = int(quantum)
        self.queues: dict[int, deque[RetainedWorkItem]] = {}
        self.roots: deque[int] = deque()
        self.active_root: int | None = None
        self.used_in_turn = 0

    def push(self, item: RetainedWorkItem) -> None:
        queue = self.queues.get(item.root_branch_id)
        if queue is None:
            queue = deque()
            self.queues[item.root_branch_id] = queue
            self.roots.append(item.root_branch_id)
        # A root may reveal several retained siblings at a context.  Stable
        # NLL/insertion ordering matches the frozen local sibling order.
        if not queue or _key(queue[-1]) <= _key(item):
            queue.append(item)
        else:
            ordered = sorted((*queue, item), key=_key)
            queue.clear(); queue.extend(ordered)

    def _discard_empty_front(self) -> None:
        while self.roots and not self.queues[self.roots[0]]:
            root = self.roots.popleft()
            self.queues.pop(root, None)
            if self.active_root == root:
                self.active_root = None
                self.used_in_turn = 0

    def pop(self) -> RetainedWorkItem | None:
        self._discard_empty_front()
        if not self.roots:
            return None
        root = self.roots[0]
        self.active_root = root
        return self.queues[root].popleft()

    def note_expansion(self, item: RetainedWorkItem) -> None:
        if self.active_root != item.root_branch_id:
            raise RuntimeError("fair policy expansion did not match selected root")
        self.used_in_turn += 1
        root = item.root_branch_id
        queue = self.queues.get(root)
        if not queue:
            self._discard_empty_front()
            self.active_root = None
            self.used_in_turn = 0
            return
        if self.used_in_turn >= self.quantum:
            if self.roots and self.roots[0] == root:
                self.roots.rotate(-1)
            self.active_root = None
            self.used_in_turn = 0

    def pending(self) -> tuple[RetainedWorkItem, ...]:
        return tuple(item for root in self.roots for item in self.queues[root])


class FairDFS:
    name: SearchOrderName = "FAIR_DFS_Q64"

    def __init__(self, quantum: int = 64) -> None:
        self._queues = _FairQueues(quantum)

    @property
    def quantum(self) -> int:
        return self._queues.quantum

    def push(self, item: RetainedWorkItem) -> None:
        self._queues.push(item)

    def pop(self) -> RetainedWorkItem | None:
        return self._queues.pop()

    def note_expansion(self, item: RetainedWorkItem) -> None:
        self._queues.note_expansion(item)

    def pending(self) -> tuple[RetainedWorkItem, ...]:
        return self._queues.pending()


class RegretBandFairDFS:
    name: SearchOrderName = "REGRET_BAND_FAIR_Q64"
    band_cycle = (0, 0, 1, 0, 2, 1, 0, 3)

    def __init__(self, quantum: int = 64) -> None:
        self._bands = {band: _FairQueues(quantum) for band in range(4)}
        self._cycle_index = 0
        self._selected_band: int | None = None

    @property
    def quantum(self) -> int:
        return self._bands[0].quantum

    def push(self, item: RetainedWorkItem) -> None:
        self._bands[item.regret_band].push(item)

    def pop(self) -> RetainedWorkItem | None:
        for offset in range(len(self.band_cycle)):
            index = (self._cycle_index + offset) % len(self.band_cycle)
            band = self.band_cycle[index]
            item = self._bands[band].pop()
            if item is not None:
                self._cycle_index = (index + 1) % len(self.band_cycle)
                self._selected_band = band
                return item
        self._selected_band = None
        return None

    def note_expansion(self, item: RetainedWorkItem) -> None:
        band = item.regret_band
        if self._selected_band != band:
            raise RuntimeError("band policy expansion did not match selected band")
        self._bands[band].note_expansion(item)
        self._selected_band = None

    def pending(self) -> tuple[RetainedWorkItem, ...]:
        return tuple(item for band in range(4) for item in self._bands[band].pending())


def make_search_order_policy(name: SearchOrderName, *, quantum: int = 64) -> SearchOrderPolicy:
    if name == "CURRENT_DFS":
        return CurrentDFS()
    if name == "FAIR_DFS_Q64":
        return FairDFS(quantum)
    if name == "REGRET_BAND_FAIR_Q64":
        return RegretBandFairDFS(quantum)
    raise ValueError(f"unknown search order policy: {name}")
