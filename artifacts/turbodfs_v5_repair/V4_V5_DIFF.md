# V4 to V5 decoder difference

V5 is a **SEARCH_SEMANTICS_EXTENSION**, not public-reference parity.

The V4 public NVARC recursion, native-token set, strict cumulative-NLL
formula, strict threshold, clock, score-only ordering, cache propagation,
candidate completion and parser remain unchanged.  V5 adds only this
pre-registered recovery: when public cumulative-NLL pruning would leave a
lane with no legal successor, restore the lowest cumulative-NLL legal
successor from the successor set considered at that exact node.  V5B, if and
only if authorized by the micro gate, changes the fixed restored count from
one to two.

No Gold, future token, sampling, threshold relaxation, budget increase,
branch-order change, or additional view is involved.  Every activation is
recorded in the cell trace as `FRONTIER_FLOOR_ACTIVATED` with the restored
rank and cumulative NLL.
