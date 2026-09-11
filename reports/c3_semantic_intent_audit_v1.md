# C3 Semantic Intent Audit V1

- 仅审计冻结 C3 checkpoint；无新模型调用、ARC grid、ARC solution 或 LLM judge。
- compile-valid: 60/60；canonical skeleton exact match: 30/60；canonical full-program exact match: 28/60。
- semantic-equivalent alternate: 3/60；semantic success: 31/60 (51.7%)。

## Semantic outcome taxonomy

- SEMANTIC_EQUIVALENT_ALTERNATE: 3
- SEMANTIC_EXACT_CANONICAL: 28
- SEMANTIC_OVER_TRANSFORMED: 19
- SEMANTIC_WRONG_OPERATION: 8
- SEMANTIC_WRONG_PARAMETER: 2

## Category semantic success

- PARAMETER_CONTRACT: 6/15 (40.0%)
- SINGLE_MACRO: 15/15 (100.0%)
- THREE_STEP_COMPOSITION: 10/15 (66.7%)
- TWO_STEP_TYPED_CHAIN: 0/15 (0.0%)

## Decision

- SEMANTIC_GROUNDING_FAILED；ARC Oracle Ladder allowed: False。
- 推荐: PROGRAM_SELECTION_REPAIR。
