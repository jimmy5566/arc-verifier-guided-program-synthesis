# P3 LDS Unit-Discrepancy V1 — Frozen Method

## Research question

Can a generic low-discrepancy traversal allocate the same retained-node budget more effectively than current DFS/fairness, without changing pruning or increasing the scientific node budget?

## Exact discrepancy definition

For every retained non-EOS successor, let its model-local ARC-token rank be r.

- r = 1: discrepancy increment = 0
- r > 1: discrepancy increment = 1

For a work item n:

D(n) = sum of these increments along its prefix.

The rank is the rank from the full legal ARC-token distribution before the frozen retention rule is applied. If frontier-floor restoration activates, the original model rank is still used; restore rank is not a discrepancy signal.

## Exact traversal

Maintain one LIFO stack S_d for each observed discrepancy count d.

1. Pop from the smallest d for which S_d is non-empty.
2. Within S_d, use depth-first LIFO traversal.
3. When one parent's retained children are inserted into the same discrepancy layer, sort them by:
   - cumulative NLL ascending,
   - insertion order ascending,
   - work-item ID ascending,
   then push in reverse so the lowest-NLL sibling is popped first.
4. A rank-1 child stays in its parent's discrepancy layer.
5. Any rank>1 child enters layer d+1.
6. No work item is expanded twice.

There is deliberately **no hard D_max**. R1024 is the fixed total scientific compute budget. This prevents the LDS experiment from silently changing retained membership or introducing a Gold-tuned discrepancy cutoff.

## Frozen non-changes

P3 must not change:

- Qwen3-4B / BF16 model semantics
- historical d24 adapter
- canonical AUG8
- cumulative-regret r=4 retention
- root-aware admission
- frontier floor = 1
- EOS handling
- candidate cap = 32
- max expanded nodes = 1024
- max_new_tokens = 931
- physical scheduler/profile policy
- cache semantics

P3 changes only which already-retained non-EOS work item is expanded next.

## Why this is a low-overfit baseline

The only new state variable is cumulative unit discrepancy. No threshold is fitted from Micro24 Gold, no task-specific condition is introduced, and no Gold-derived feature is available during generation.

The Gold diagnostic motivated the hypothesis class, but does not determine the ordering parameters.
