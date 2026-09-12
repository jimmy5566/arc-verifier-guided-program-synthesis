# DOWNSTREAM Capability Gap Forensics V1

这是一个 solution-aware、development-only 的离线能力取证；没有运行 Grid Recognizer、Qwen 或新推理。公开产物不含任务 ID、ARC grids、solutions、oracle programs 或 test outputs。

## 冻结 cohort

- Tasks: 30 development
- Cohort commitment hash: `0a0e6fe94f87f5b04babb687dbeaa5624232ba50c3581e4a736cebe355d30e32`
- Frozen DOWNSTREAM_V1 candidate-space hash: `3016ac7e5dd1cbaf8fc5a98c00916482443d65b8af4127b4f8f22684ec687b52`

## 结果

- Representable: **0/30 = 0.0%**
- Search found oracle/equivalent: **0**
- Search missed: **0**
- Compiler gap: **0**
- Capability gap: **30**

| 分类 | Tasks |
| --- | ---: |
| REPRESENTABLE_EXISTING | 0 |
| REPRESENTABLE_BUT_SEARCH_MISSED | 0 |
| DSL_VALID_BUT_COMPILER_UNSUPPORTED | 0 |
| MISSING_PRIMITIVE | 0 |
| MISSING_RELATIONAL_CAPABILITY | 8 |
| MISSING_COMPOSITION | 2 |
| MISSING_OUTPUT_CONSTRUCTION | 1 |
| MISSING_ITERATION_OR_REPEAT | 15 |
| MISSING_CONDITIONAL_OR_ROLE_LOGIC | 4 |
| MISSING_PATTERN_COMPLETION | 0 |
| OTHER_CAPABILITY_GAP | 0 |

## Top missing capabilities

| 缺口分类 | Tasks |
| --- | ---: |
| MISSING_ITERATION_OR_REPEAT | 15 |
| MISSING_RELATIONAL_CAPABILITY | 8 |
| MISSING_CONDITIONAL_OR_ROLE_LOGIC | 4 |
| MISSING_COMPOSITION | 2 |
| MISSING_OUTPUT_CONSTRUCTION | 1 |

## 历史 4 个 apparent train-consistent/test-wrong

历史私有记录中有 4 个 train-consistent 候选。重新用逐个 `numpy.array_equal` 比较 JSON-normalized grids 后，发现 4 个是 tuple/list 容器比较造成的计分假阴性，真实 test-wrong 为 0。因此它们不是 coordinate/color/size/object-identity overfit 证据，也不支持据此引入 ranking 或 generalization 修改。

## 决策

主瓶颈：**CAPABILITY_COVERAGE**。

建议的唯一下一实验：**CAPABILITY_LIBRARY_EXPANSION_V1**。

## 完整性与泄漏审计

- solutions 仅在 `recognition.downstream_capability_gap_forensics` 的离线 oracle 边界读取。
- 30 题 cohort 在读取 solutions 前按 development split 和 SHA-256 固定；不会按 solver 成功筛选。
- 全部 representable 结论均要求 schema → type → parameter → compiler → executor → train exact → test exact。
- 无 task-ID production branch；私有 task IDs、program fingerprints 及任何逐题解释位于 ignored `artifacts/`。
