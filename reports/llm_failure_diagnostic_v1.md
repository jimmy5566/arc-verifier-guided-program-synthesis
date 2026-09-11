# LLM Failure Diagnostic V1

## Static checkpoint audit

The full 1000-task continuation is disabled. This script did not load solutions. Historical checkpoints did not preserve raw responses, parsed hypotheses, primitive IDs, or program depth; those missing values are reported as unavailable rather than reconstructed.

### development_confirmation

- total_tasks: 100/100 (100.00%)
- parseable_response: 88/100 (88.00%)
- schema_valid_program: 88/100 (88.00%)
- type_valid_program: 28/100 (28.00%)
- executable_program: 1/100 (1.00%)
- train_consistent_program: 0/100 (0.00%)
- frozen_test_prediction: 0/100 (0.00%)
- exact_test_correct: 0/100 (0.00%)

### held_out

- total_tasks: 257/257 (100.00%)
- parseable_response: 216/257 (84.05%)
- schema_valid_program: 216/257 (84.05%)
- type_valid_program: 74/257 (28.79%)
- executable_program: 1/257 (0.39%)
- train_consistent_program: 0/257 (0.00%)
- frozen_test_prediction: 0/257 (0.00%)
- exact_test_correct: 0/257 (0.00%)

## Controlled retrieval diagnostic

| Condition | Median prompt tokens | Exact | Train-consistent rate | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| full | 6976.5 | 0 | 0.000 | 306.1 |
| top15 | 2216.0 | 0 | 0.000 | 354.0 |
| top30 | 3825.5 | 0 | 0.000 | 220.2 |

Hierarchical retrieval promising: **False**.

## Evidence-based conclusion

Primary bottleneck: **Capability API comprehension / parameter inference bottleneck**.

Exactly one recommended next experiment: **LLM_PROGRAM_SYNTHESIS_V2**.
