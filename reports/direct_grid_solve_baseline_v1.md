# DIRECT GRID SOLVE BASELINE V1

D0 使用 Qwen3-8B，只读取完整 train input/output grids 与 test input；每题一次贪心调用，输出至多两个网格候选。Kaggle 推理未导入 target、oracle、Semantic IR、capability library、compiler、executor、verifier 或 search；本地仅在预测文件冻结后进行 exact scoring。

## Direct 指标

- valid_grid_rate: **66.7%**
- top1_test_exact: **3.3%**
- any_of_2_test_exact: **3.3%**
- output_shape_correct: **46.7%**
- color_set_valid: **66.7%**
- mean_cell_accuracy: **33.3%**
- mean_changed_cell_precision: **26.3%**
- mean_changed_cell_recall: **22.2%**

## 与 Recognition V1 的逐题交集

- recognition_success_direct_exact: 0
- recognition_success_direct_fail: 4
- recognition_fail_direct_exact: 1
- recognition_fail_direct_fail: 25

## 解释

- 状态：**DIRECT_SOLVER_VERY_WEAK**
- Semantic IR 可能形成接口限制的信号：**PARTIAL**
- 相对冻结 executable capability coverage（1/30）的 direct-only exact 增益：**+1 task(s)**

- Kaggle kernel wall runtime：**596.8s**；所有 worker generation 时间之和：**1260.5s**。

这是本地 frozen 30-task development exact baseline，不能等同于 Kaggle Public LB 百分比，也不构成模型能力上限。

## Failure taxonomy

- INVALID_GRID_OUTPUT: 10
- MISSED_ITERATION: 2
- OTHER_GRID_ERROR: 1
- OVER_TRANSFORM: 1
- PARTIAL_TRANSFORMATION: 1
- RIGHT_SHAPE_WRONG_COLOR: 2
- WRONG_CONDITIONAL_RULE: 2
- WRONG_DIMENSION: 6
- WRONG_ITERATION_COUNT: 4

公开聚合结果不包含 task IDs、grids、targets、raw responses 或 oracle records。
