# V7、R1 与家族平衡损失处理的配对分析

| 家族 | V7 | R1 | 家族平衡 | V7→R1 净修复 | V7→家族平衡 净修复 | R1→家族平衡 净修复 | 分类 |
|---|---:|---:|---:|---:|---:|---:|---|
| color mapping | 16/16 | 16/16 | 16/16 | 0 | 0 | 0 | UNCHANGED |
| complete missing structure | 16/16 | 16/16 | 16/16 | 0 | 0 | 0 | UNCHANGED |
| conditional action | 2/24 | 5/24 | 1/24 | 3 | -1 | -4 | IMPROVED_THEN_REGRESSED |
| connected components | 0/12 | 1/12 | 1/12 | 1 | 1 | 0 | IMPROVED_STABLY |
| difference | 2/12 | 1/12 | 2/12 | -1 | 0 | 1 | RECOVERED_IN_FAMILY_BALANCED |
| inside/contains | 0/12 | 3/12 | 0/12 | 3 | 0 | -3 | IMPROVED_THEN_REGRESSED |
| mask set | 2/24 | 2/24 | 4/24 | 0 | 2 | 2 | MIXED_HIGH_CHURN |
| orientation | 5/12 | 5/12 | 4/12 | 0 | -1 | -1 | MIXED_HIGH_CHURN |
| overlay | 4/16 | 4/16 | 4/16 | 0 | 0 | 0 | UNCHANGED |
| propagation | 2/16 | 3/16 | 1/16 | 1 | -1 | -2 | IMPROVED_THEN_REGRESSED |
| recolor | 28/28 | 28/28 | 28/28 | 0 | 0 | 0 | UNCHANGED |
| relation selector action | 3/24 | 3/24 | 2/24 | 0 | -1 | -1 | MIXED_HIGH_CHURN |
| rotate | 16/16 | 9/16 | 16/16 | -7 | 0 | 7 | RECOVERED_IN_FAMILY_BALANCED |
| same color | 12/12 | 9/12 | 12/12 | -3 | 0 | 3 | RECOVERED_IN_FAMILY_BALANCED |
| selector prerequisites | 36/36 | 36/36 | 36/36 | 0 | 0 | 0 | UNCHANGED |
| width | 2/12 | 0/12 | 0/12 | -2 | -2 | 0 | REGRESSED_STABLY |

仅报告配对结果；不将结果解释为梯度或因果机制。
