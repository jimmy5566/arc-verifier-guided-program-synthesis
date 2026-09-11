# Contextual Parameter Reasoning V1

- 60-case frozen Macro API benchmark；无 ARC data、ARC solution 或新的 ARC inference。
- R1/R2/R3 的全部程序冻结后，父进程才使用 canonical semantic scorer 做评分。

| Condition | Success | Remaining repaired / 12 | Q1 regression / 36 | Patches | Precision |
| --- | ---: | ---: | ---: | ---: | --- |
| R1_RELATIONAL_NORMALIZER | 48/60 (80.0%) | 12/12 | 0/36 | 60 | 50.0% |
| R2_CONTEXTUAL_CANDIDATE_CLASSIFIER | 18/60 (30.0%) | 0/12 | 18/36 | 37 | 13.5% |
| R3_CONSERVATIVE_CONTEXTUAL_REPAIR | 33/60 (55.0%) | 12/12 | 15/36 | 86 | 34.9% |

## 最佳条件

- best: `R1_RELATIONAL_NORMALIZER`；status: `CONTEXTUAL_PARAMETER_LAYER_SOLVED`。
- 60.0% → 80.0%；新增解决 12；距离 80% parameter-only ceiling 尚 0 cases。
- remaining repair: 12/12；regression: 0/36；abstain: 36；compile-valid is retained by frozen skeleton/local patch contract。

## 字段与转移

- transitions: `{'SEMANTIC_EQUIVALENT_ALTERNATE_TO_SEMANTIC_EQUIVALENT_ALTERNATE': 3, 'SEMANTIC_EXACT_CANONICAL_TO_SEMANTIC_EXACT_CANONICAL': 33, 'SEMANTIC_OVER_TRANSFORMED_TO_SEMANTIC_OVER_TRANSFORMED': 3, 'SEMANTIC_PARTIAL_TO_SEMANTIC_PARTIAL': 6, 'SEMANTIC_WRONG_OPERATION_TO_SEMANTIC_WRONG_OPERATION': 3, 'SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EXACT_CANONICAL': 12}`
- per field: `{'color_source': {'remaining_after': 0, 'remaining_before': 6, 'repaired': 6}, 'direction': {'remaining_after': 0, 'remaining_before': 5, 'repaired': 5}, 'orientation': {'remaining_after': 0, 'remaining_before': 12, 'repaired': 12}, 'shape_source': {'remaining_after': 0, 'remaining_before': 7, 'repaired': 7}}`

## 运行

- worker→GPU: `{'0': 0, '1': 1}`；wall: 507.8s。
- warmup: `{'bytes_read': 16381516776, 'event': 'MODEL_FILES_WARMED', 'model_path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1', 'seconds': 109.10294839799997, 'shard_count': 5}`

## 泄漏审计

- model input 仅含 instruction、family、frozen skeleton、target slot、legal candidates、relation features；不含 case ID、canonical、semantic label 或 expected candidate。
- R1 映射由全局 Macro contract 和 relation ontology 驱动；R3 默认 KEEP Q1。
