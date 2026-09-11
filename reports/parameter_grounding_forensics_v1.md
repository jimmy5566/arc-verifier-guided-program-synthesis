# Parameter Grounding Forensics V1

- 仅分析冻结 S2 checkpoint；无新模型调用、ARC 数据、ARC solution 或 LLM judge。
- S2 wrong-parameter: 17；pure parameter failures: 17；structural confounds: 0。
- C3 static-slot interface confounds: 17；single/multiple parameter errors: 0/17。

## Parameter fields

- color_source: 3
- count_kind: 1
- direction: 8
- operation: 2
- orientation: 15
- shape_source: 7

## Error subtypes

- MISSING_SEMANTIC_SLOT: 3
- WRONG_COUNT_SEMANTICS: 1
- WRONG_ENUM_VALUE: 2
- WRONG_SYMBOLIC_SOURCE: 30

## Interpretation

- 这里的 structural confound 仅指 family 或 Macro-id skeleton 不兼容；C3 static-slot interface mismatch 单独报告，不能被误写为 skeleton 选择错误。
