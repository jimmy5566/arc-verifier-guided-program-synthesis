# Parameter Semantic Retrieval V1

- 冻结 60-case Macro API benchmark；没有 ARC 数据、ARC solution 或新 ARC inference。
- 模型仅执行 local Qwen3-8B teacher-forced forward likelihood；无 completion generation、JSON schema 解析或自由参数生成。
- 所有 Q1–Q4 预测冻结后，父进程才读取 semantic scorer/canonical program 进行离线评分；repair gate 从不接收它们。

## Theoretical limit

- parameter-only ceiling: 48/60 = 80.0%。

## Conditions

| Condition | Semantic Success | Repaired / 17 | Patches | Patch precision | Regression / 31 |
| --- | --- | ---: | ---: | --- | ---: |
| Q0 S2 frozen | 31/60 (51.7%) | 0/17 | 0 | N/A | N/A |
| Q1_DETERMINISTIC_ONTOLOGY_REPAIR | 36/60 (60.0%) | 5/17 | 33 | 36.4% | 0/31 |
| Q2_LIKELIHOOD_PARAMETER_REPAIR | 15/60 (25.0%) | 0/17 | 59 | 23.7% | 16/31 |
| Q3_CONSENSUS_PARAMETER_REPAIR | 25/60 (41.7%) | 0/17 | 40 | 25.0% | 6/31 |
| Q4_CONSERVATIVE_HYBRID_REPAIR | 28/60 (46.7%) | 3/17 | 50 | 34.0% | 6/31 |

## Q4 Conservative Hybrid

- wrong-parameter remaining: 13；patch count: 50；correct patches: 17；incorrect patches: 33；abstain failure cases: 0。
- repair coverage: 100.0%；S2 success preserved: 25/31；compile-valid: 53/60 (88.3%)。

## Per-slot Q4

| Field | S2 errors | Patches | Correct patches | Remaining errors | Repair rate |
| --- | ---: | ---: | ---: | ---: | --- |
| color_source | 3 | 3 | 3 | 0 | 100.0% |
| count_kind | 1 | 1 | 1 | 0 | 100.0% |
| direction | 8 | 8 | 8 | 0 | 100.0% |
| operation | 2 | 2 | 2 | 0 | 100.0% |
| orientation | 15 | 5 | 3 | 12 | 20.0% |
| shape_source | 7 | 0 | 0 | 7 | 0.0% |

## Orientation

- candidate accuracy proxy (correct patch recall): 20.0%；repair precision: 60.0%；recall: 20.0%。

## Decision

- best source: Q1_DETERMINISTIC_ONTOLOGY_REPAIR；strict-schema problem solved: 是（本实验无自由 JSON schema）。
- SelectiveParameterRepairV2: 部分；status: PARAMETER_LAYER_STILL_BLOCKED。
- overall semantic success: 51.7% → 60.0%；reached 80% ceiling: NO。
- worker→GPU: {'0': 0, '1': 1, '2': 2, '3': 3}；runtime: 823.3s。
