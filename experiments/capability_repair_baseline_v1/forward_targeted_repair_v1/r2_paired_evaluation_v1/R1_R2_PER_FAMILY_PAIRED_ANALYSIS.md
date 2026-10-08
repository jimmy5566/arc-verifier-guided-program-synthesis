# R1 / R2 逐家族配对分析

此分析只读取冻结的 V7、R1 和 R2 逐样本预测；不改变既有结果，不访问 FINAL_AUDIT，不启动 R3。

- 配对分析 JSON SHA256：`6dc3ba24762484d6514a35f51d6297461aa44efdc32dded060b68e52563f5283`
- 记号：`修复/伤害/净值` 分别为 n01/n10/(n01−n10)。

| 组别 | 家族 | n | V7 | R1 | R2 | R1−V7 修复/伤害/净值 | R2−V7 修复/伤害/净值 | R2−R1 修复/伤害/净值 | 分类 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| ATOMIC | connected components | 12 | 0 | 1 | 1 | 1/0/+1 | 1/0/+1 | 1/1/+0 | IMPROVED_STABLY |
| ATOMIC | difference | 12 | 2 | 1 | 0 | 1/2/-1 | 0/2/-2 | 0/1/-1 | REGRESSED_STABLY |
| ATOMIC | inside/contains | 12 | 0 | 3 | 3 | 3/0/+3 | 3/0/+3 | 1/1/+0 | IMPROVED_STABLY |
| ATOMIC | orientation | 12 | 5 | 5 | 4 | 1/1/+0 | 1/2/-1 | 1/2/-1 | REGRESSED_STABLY |
| ATOMIC | recolor | 12 | 12 | 12 | 12 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| ATOMIC | same color | 12 | 12 | 9 | 6 | 0/3/-3 | 0/6/-6 | 1/4/-3 | REGRESSED_STABLY |
| ATOMIC | selector prerequisites | 12 | 12 | 12 | 12 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| ATOMIC | width | 12 | 2 | 0 | 1 | 0/2/-2 | 0/1/-1 | 1/0/+1 | MIXED_HIGH_CHURN |
| COMPOSITION | conditional action | 24 | 2 | 5 | 5 | 3/0/+3 | 4/1/+3 | 3/3/+0 | IMPROVED_STABLY |
| COMPOSITION | mask set | 24 | 2 | 2 | 4 | 2/2/+0 | 4/2/+2 | 4/2/+2 | MIXED_HIGH_CHURN |
| COMPOSITION | relation selector action | 24 | 3 | 3 | 2 | 2/2/+0 | 2/3/-1 | 1/2/-1 | REGRESSED_STABLY |
| COMPOSITION | selector prerequisites | 24 | 24 | 24 | 24 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| RETENTION | color mapping | 16 | 16 | 16 | 16 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| RETENTION | complete missing structure | 16 | 16 | 16 | 16 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| RETENTION | overlay | 16 | 4 | 4 | 1 | 3/3/+0 | 1/4/-3 | 1/4/-3 | REGRESSED_STABLY |
| RETENTION | propagation | 16 | 2 | 3 | 3 | 3/2/+1 | 2/1/+1 | 3/3/+0 | IMPROVED_STABLY |
| RETENTION | recolor | 16 | 16 | 16 | 16 | 0/0/+0 | 0/0/+0 | 0/0/+0 | UNCHANGED |
| RETENTION | rotate | 16 | 16 | 9 | 16 | 0/7/-7 | 0/0/+0 | 7/0/+7 | RECOVERED_IN_R2 |

## 干扰解释

- R1 的组成能力净收益来自：conditional action。
- R1 伤害的原子能力：difference, same color, width。
- R2 相比 R1 进一步伤害的原子能力：difference, orientation, same color。
- R1→R2 恢复的保持能力：rotate。
- R2 仍低于 V7 的保持能力：overlay。
- 同一示例在轮次间反复翻转的家族：overlay, propagation, rotate, conditional action, mask set, relation selector action, connected components, difference, inside/contains, orientation, same color, width。

分类基于同一冻结 episode 集的逐例正确性转换；它描述训练干扰，不构成任何 R3 启动授权。
