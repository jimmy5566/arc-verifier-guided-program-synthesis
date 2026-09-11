# Contextual Parameter Forensics V1

- 完全离线 Q1 后验审计；未调用模型，未使用 ARC 数据。
- Q1 remaining pure parameter failures: 12。

## 分布

- primary taxonomy: `{'MULTI_SLOT_DEPENDENCY': 12}`
- aspects: `{'CONTEXT_REQUIRED': 12, 'IMPLICIT_DIRECTION': 5, 'IMPLICIT_ORIENTATION': 12, 'IMPLICIT_SOURCE_REFERENCE': 5, 'MULTI_SLOT_DEPENDENCY': 12, 'RELATIONAL_INFERENCE': 12}`
- fields: `{'color_source': 1, 'count_kind': 1, 'direction': 5, 'orientation': 12, 'shape_source': 7}`
- lexical / relational / contextual: `{'lexical': 0, 'relational': 12, 'contextual': 12}`
- truly repairable: 12；ambiguous: 0。

## 结论

- path 链失败主要是 source-wrapper 的隐式关系；count→generate 链主要是跨 slot 的 contract-scoped 参数关系。
- 后续 R1 仅使用全局 Macro contract 与 typed relation features；R2/R3 不接收 canonical 或 failure labels。
