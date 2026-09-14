# Public-reference Native selection/search audit

This audit compares the current Native Qwen3-4B grid-transduction pipeline with public NVARC reference mechanisms before the A/B/C/D diagnostic.  The implementation is independent: only algorithmic ideas are used, with provenance frozen in `configs/PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json`.

| Area | Current Native baseline A | Public-reference mechanism | A/B/C/D decision |
|---|---|---|---|
| Input representation | Native 16-token digit-row format; reversible geometric/color/order views | Native digit-grid format and augmented views | Keep unchanged |
| Candidate creation | One deterministic greedy completion per one of 32 fixed views | Bounded token-probability DFS / multiple decoded branches | C: deterministic bounded beam branches, 4 per view |
| Search budget | 32 raw candidates/task before parse/dedup | Per-puzzle bounded decoding time/state budget | C: fixed 128 raw branches/task, then parse/dedup |
| Output deduplication | Exact full multi-test output deduplication | Group equivalent decoded outputs | Keep; retain support count for C |
| Selection | Mean original-prompt conditional log likelihood | Equivalent-output support plus augmentation score (KGMoN-style) | B/D: support minus mean fixed-view NLL |
| Second submission | Not a separate declared protocol | Ordered distinct output list | B/D: next distinct output in the predeclared ranking |
| Adaptation | None | Public notebook also uses task LoRA | Explicitly excluded: no TTT/LoRA |
| GPU topology | Four independent worker processes already supported | Public code uses a different runtime | C keeps one complete 4B model per L4; no sharding |

## Highest-value gaps

1. The baseline did not branch within a prompt: candidate diversity came only from static reversible views.  C adds a strictly bounded deterministic decoding branch search.
2. The baseline ranked each deduplicated output independently.  B/D evaluate model-generated grids under fixed reversible views, then aggregate equivalent outputs before selecting two distinct attempts.
3. The historical A artifact dropped pre-dedup duplicate provenance.  B cannot reconstruct support that was not persisted, so it records support as one rather than inventing it.  C persists actual support counts, allowing D to use the full aggregation signal.

## Safeguards

- No solution file is accepted by A/B/C/D generation or reranking commands.
- C is task-level crash-safe/resumable and has a fixed 32 x 4 raw-branch budget.
- Exact scoring is a separate post-freeze operation.
- No task identifier, target grid, RuleSpec, router, SOAR, or adaptation logic occurs in this experiment.
