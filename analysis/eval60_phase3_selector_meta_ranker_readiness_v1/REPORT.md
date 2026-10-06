# Final-Candidate Meta-Ranker Data Readiness V1

Classification: `META_RANKER_DATA_NOT_READY`.

## MEASURED

- The frozen current Phase-3 dense DFS archive covers 60 tasks / 89 outputs at d24 and d48 and passes source-completeness audits.
- The frozen union oracle contains Gold on 35/89 outputs.
- The selector diagnostic cohort therefore contains 35 outputs from 27 tasks and is explicitly post-freeze Gold-derived / exposed development.
- The current S2 selector evidence stack was built on those 35 oracle-hit outputs.
- Grouped task-level CV is technically possible inside the 35-output cohort, but that cohort was selected because Gold is known to be present.
- Historical alternatives are not equivalent final-candidate datasets:
  - Nonblind LOO Transfer30 is a TTT depth/view routing distribution.
  - Earlier native/greedy ranker data comes from sparse historical candidate distributions.
  - Search-order node separability labels Gold-prefix search-tree nodes, not final candidate correctness.
  - Expanded-sibling/frontier datasets are search-choice datasets, not final candidate-ranking groups.

## INFERRED

A model trained only on the exposed 35-output oracle-hit cohort could be useful for exploratory diagnostics, but grouped CV there cannot establish generalization to unseen dense-DFS outputs. Mixing incompatible historical datasets would change the learning problem rather than solve the data-readiness issue.

## NOT ESTABLISHED

- This does not show that supervised final-candidate ranking cannot work.
- It does not authorize new GPU candidate generation.
- It does not justify reporting a model trained on the exposed 35 outputs as a validated selector.

## Decision

Stop before model fitting and classify `META_RANKER_DATA_NOT_READY`.

Any future supervised selector requires a separately designed source of comparable labeled candidate groups that is not selected from the same exposed Gold-hit cohort. New GPU work to construct such a dataset requires a separate preregistered decision.
