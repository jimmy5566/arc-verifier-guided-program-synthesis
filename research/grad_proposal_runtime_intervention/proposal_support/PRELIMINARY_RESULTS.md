# Preliminary results for proposal use

## Evidence label

All numbers below are **controlled preliminary evidence**. Frozen ARC results
are motivation for responsibility mismatch; Stages 1--3 are synthetic CPU-only
environments. They are not real interactive-LLM benchmark results.

## Frozen60 ARC motivation

| Metric | Result |
|---|---:|
| Any-of-K | 30/60 |
| Top-1 | 16/60 |
| Top-2 | 21/60 |
| Generator-limited failures | 30 |
| Selection failures | 14 |
| Non-selection-responsible failures | 30/44 = 68.2% |
| Conditional selector accuracy | 16/30 = 53.3% |

## Stage-2C: heterogeneous runtime intervention

ID BASE controlled utility:

| Policy | Utility |
|---|---:|
| Attribution-aware | .675 |
| Uncertainty-only | .463 |
| Strongest simple non-attribution baseline | .513 |
| Shuffled attribution | .426 |
| Oracle attribution | .810 |

## Stage-3: ambiguous multi-fault intervention

ID, MEDIUM cost:

| Budget | Attribution-aware | Uncertainty-only | Strongest simple | Shuffled | Oracle |
|---|---:|---:|---:|---:|---:|
| 1 | .519 | .336 | .406 | .362 | .586 |
| 2 | .589 | .337 | .446 | .374 | .668 |

OOD dual-fault shift, Budget 2:

| Attribution-aware | Uncertainty-only | Strongest simple | Oracle |
|---:|---:|---:|---:|
| .471 | .184 | .338 | .561 |

Interpretation: knowing *what is probably wrong* improved the allocation of
heterogeneous, costly interventions beyond merely knowing that an episode was
uncertain or likely failed. This is a proposal-supporting mechanism result,
not a final thesis result.
