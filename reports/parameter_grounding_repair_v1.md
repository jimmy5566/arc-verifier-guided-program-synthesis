# Parameter Grounding Repair V1

- 冻结的 60-case Macro API benchmark；无 ARC 数据、ARC solution 或新 ARC inference。
- P0 仅引用冻结 S2；P1–P4 在同一份预冻结配置下各运行一次。family、skeleton、Macro ID 与 step order 对每个 case 均冻结为 S2 选择。
- worker 只收到 instruction、冻结 S2 family/skeleton/program 和 registry-derived parameter contract；canonical program 与 semantic label 仅在四个 prediction 全部冻结后由父进程评分。

## Phase A forensics

- S2 wrong-parameter: 17；pure parameter failures: 17；non-parameter structural failures: 0。
- C3 static-slot interface confounds: 17。

## Conditions

| Condition | Semantic Success | Wrong Parameter | Repaired / 17 | Regression / 31 | Compile | Calls |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| P0 S2 frozen | 31/60 (51.7%) | 17 | 0 | 0 | 60/60 (100.0%) | 0 |
| P1_NORMALIZED_SLOT_EXTRACTION | 3/60 (5.0%) | 7 | 0 | 28 | 13/60 (21.7%) | 60 |
| P2_CONSTRAINED_PARAMETER_CHOICE | 0/60 (0.0%) | 0 | 0 | 31 | 0/60 (0.0%) | 60 |
| P3_SLOTWISE_PARAMETER_SELECTION | 0/60 (0.0%) | 0 | 0 | 31 | 0/60 (0.0%) | 215 |
| P4_SELECTIVE_PARAMETER_REPAIR | 31/60 (51.7%) | 17 | 0 | 0 | 60/60 (100.0%) | 60 |

## P4 field repair

| Parameter | S2 errors | P4 repaired | Repair rate |
| --- | ---: | ---: | --- |
| color_source | 3 | 0 | 0.0% |
| count_kind | 1 | 0 | 0.0% |
| direction | 8 | 0 | 0.0% |
| operation | 2 | 0 | 0.0% |
| orientation | 15 | 0 | 0.0% |
| shape_source | 7 | 0 | 0.0% |

## P4 transition focus

- SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EXACT_CANONICAL: 0
- SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EQUIVALENT_ALTERNATE: 0
- SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_WRONG_PARAMETER: 17
- SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_WRONG_OPERATION: 0
- SEMANTIC_EXACT_CANONICAL_TO_SEMANTIC_EXACT_CANONICAL: 28
- SEMANTIC_EXACT_CANONICAL_TO_SEMANTIC_WRONG_PARAMETER: 0

## Decision

- best: P4_SELECTIVE_PARAMETER_REPAIR — repair 0/17 (0.0%); regression 0/31; compile 100.0%。
- status: PARAMETER_LAYER_STILL_BLOCKED；SelectiveParameterRepairV1 enabled: False。
- worker→GPU: {'0': 0, '1': 1, '2': 2, '3': 3}；runtime: 416.3s。
