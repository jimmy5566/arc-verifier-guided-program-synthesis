# TurboDFS V3 mechanism audit

The V2 `SEARCH_EXHAUSTED` micro failures must not be treated as a reason to
increase search time or a forward-pass budget.  The public notebook at NVARC
commit `846d0198efa752534594e321fc3289fc0a06c657`, notebook
`ARC-AGI1/002_ivan_arc1.ipynb`, was rechecked against the local implementation.

## Cumulative-NLL parity

The public code forms `nll = prior_score - logits.float().cpu().log_softmax(-1)`
and retains a token exactly when `score < -log(0.2)`.  It uses cumulative,
absolute NLL: no mean-length normalization, relative-frontier normalization,
or prefix-length adjustment.  The local V2 computation and strict threshold
are equivalent; the local `next_score >= threshold` rejection is the logical
complement of the public strict `<` keep rule.  The public code does not offer
a guarantee that one lane survives pruning, and its compact native ARC token
filter can legitimately empty a frontier.

## Identified local/public difference and V3 change

The local V1/V2 implementation applied a non-public cap of four *retained
completed suffixes per prompt*, including during recursive suffix propagation.
The public implementation keeps every completed suffix.  While this cap was
not observed as the terminal flag on the two V2 zero-complete micro cells, it
is the only remaining local frontier-loss mechanism that differs from the
verified public decoder.  V3 removes that cap (`null`) and retains all completed
suffixes exactly as the public recursive code does.  This is a public-parity
frontier-preservation correction, not a time, branch, node, sampling, beam, or
Gold-informed budget increase.

The cumulative NLL threshold itself is deliberately unchanged.  If V3 still
exhausts, that outcome is consistent with the public strict pruning rule and
does not justify V4 tonight.

## Frozen scope

V3 retains: the public 540-second time stop, no forward-pass cap, compact
native-token vocabulary, float32 CPU log-softmax, score-first ordering, KV
reuse, deterministic views, and target-blind calibration/block ordering.
It adds no Gold access or correctness-conditioned scheduling.
