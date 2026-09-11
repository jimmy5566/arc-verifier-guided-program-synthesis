# Program Selection Repair V1

- 冻结的 60-case Macro API semantic-interface benchmark；没有 ARC 数据、ARC solution 或新 ARC inference。
- S0 仅引用冻结 C3 审计；S1–S4 在同一份预冻结配置下各执行一次。
- 评分继续使用 deterministic C3 semantic scorer；没有 LLM judge。公开结果不含 raw model response 或 program material。

## 总表

| Condition | Semantic Success | Exact Canonical | Equivalent Alternate | Over-Transform | Wrong Operation | Wrong Parameter | Compile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 C3 frozen | 31/60 (51.7%) | 28 | 3 | 19 | 8 | 2 | 60/60 (100.0%) |
| S1_MINIMALITY_AWARE_SELECTION | 28/60 (46.7%) | 28 | 0 | 13 | 6 | 4 | 55/60 (91.7%) |
| S2_FAMILY_FIRST_SELECTION | 31/60 (51.7%) | 28 | 3 | 3 | 3 | 17 | 60/60 (100.0%) |
| S3_CONTRASTIVE_SKELETON_RANKING | 15/60 (25.0%) | 15 | 0 | 0 | 26 | 19 | 60/60 (100.0%) |
| S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY | 24/60 (40.0%) | 24 | 0 | 0 | 10 | 26 | 60/60 (100.0%) |

## 选择与回归

### S1_MINIMALITY_AWARE_SELECTION

- S0 OVER_TRANSFORMED → success: 0
- S0 WRONG_OPERATION → success: 0
- S0 WRONG_PARAMETER → success: 0
- S0 success retained / regressed: 28 / 3 (9.7%)
- generation calls: 60; selected extra semantic operations: 27

### S2_FAMILY_FIRST_SELECTION

- S0 OVER_TRANSFORMED → success: 0
- S0 WRONG_OPERATION → success: 0
- S0 WRONG_PARAMETER → success: 0
- S0 success retained / regressed: 31 / 0 (0.0%)
- generation calls: 120; selected extra semantic operations: 23

### S3_CONTRASTIVE_SKELETON_RANKING

- S0 OVER_TRANSFORMED → success: 0
- S0 WRONG_OPERATION → success: 0
- S0 WRONG_PARAMETER → success: 0
- S0 success retained / regressed: 15 / 16 (51.6%)
- generation calls: 120; selected extra semantic operations: 0

### S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY

- S0 OVER_TRANSFORMED → success: 0
- S0 WRONG_OPERATION → success: 0
- S0 WRONG_PARAMETER → success: 0
- S0 success retained / regressed: 24 / 7 (22.6%)
- generation calls: 180; selected extra semantic operations: 1

## 决策

- best: S2_FAMILY_FIRST_SELECTION — 31/60 (51.7%); compile 100.0%。
- semantic status: SEMANTIC_GROUNDING_FAILED；ARC_ORACLE_LADDER_READY: NO。
- worker→GPU: {'0': 0, '1': 1, '2': 2, '3': 3}；runtime: 432.0s。
