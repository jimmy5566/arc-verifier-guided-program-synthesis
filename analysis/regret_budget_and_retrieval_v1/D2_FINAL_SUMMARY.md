# D2 final post-freeze summary

**Scope:** nonblind development, post-freeze Gold diagnosis only. Historical union remains **33/89 unchanged**.

## Integrity

- Frozen generation reused: 39/39; raw hashes matched before and after CPU scoring.
- GPU rerun: NO; scientific configuration changed: NO.
- Actual solutions schema: 120-entry task-ID dictionary whose values are compact output-grid lists. The strict normalizer also supports a top-level list only when mapped by the exact frozen challenge ordering.

## Regret budget response

- R512/R1024/R2048/R4096: 0 / 2 / 2 / 6 (16 validation cells).
- New Regret Gold rescues: 6/16 (37.5%).
- Rescue IDs: d59b0160:o0:d24:identity, 97d7923e:o0:d48:identity, 97d7923e:o0:d24:flip_ud, d59b0160:o0:d24:flip_ud, d59b0160:o0:d24:transpose, 20270e3b:o0:d12:identity.
- First-Gold node distribution: <=512=0, 513-1024=2, 1025-2048=0, 2049-4096=4.
- Prefix budget-reuse validity: YES (8 checked rows).
- Recommended Regret budget: 4096.
- Regret runaway cause: SEARCH_TREE_EXPANSION.

## Retrieval traces

### TOPK2

- Gold hits: 0/6.
- Failures: BRANCH_NOT_DISCOVERED=0, FRONTIER_STARVATION=0, CANDIDATE_CAP_PREEMPTION=0, NODE_BUDGET_PREEMPTION=0, OTHER=6.
- Dominant classification: NOT_ESTABLISHED.

### AFFINE

- Gold hits: 0/2.
- Failures: BRANCH_NOT_DISCOVERED=0, FRONTIER_STARVATION=0, CANDIDATE_CAP_PREEMPTION=0, NODE_BUDGET_PREEMPTION=0, OTHER=2.
- Dominant classification: NOT_ESTABLISHED.

## Decision

- NEED_COMPLEX_EARLY_STOP: YES.
- NEED_BUDGET_ROUTER: YES.
- Next: validate a budget-aware/early-stop control before primary use; do not promote this development diagnostic directly.
