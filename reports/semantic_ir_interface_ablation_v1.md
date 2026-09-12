# Semantic IR Interface Ablation V1

这是冻结的 30-task development-only Recognition interface 实验。C1/C2/C3 在 Kaggle 4×L4 上生成、冻结并哈希后，才在本地隔离 oracle 层评分；C0 复用 V1 的冻结预测。输入、模型、semantic ontology、temperature、top-1 和无 retry 规则均固定。

| Condition | Interface | Schema valid | Family | Core op | Role F1 | Relation F1 | Semantic success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | full JSON (reused) | 66.7% | 16.7% | 36.7% | 9.8% | 13.3% | 4/30 (13.3%) |
| C1 | flat typed slots | 86.7% | 10.0% | 16.7% | 31.3% | 16.7% | 1/30 (3.3%) |
| C2 | finite candidate slots | 70.0% | 6.7% | 6.7% | 28.3% | 10.0% | 0/30 (0.0%) |
| C3 | family then slots | 100.0% | 10.0% | 36.7% | 43.0% | 13.3% | 0/30 (0.0%) |

`INTERFACE_RECOVERY`（C0 failure 而替代 interface success）为 C1=0、C2=0、C3=0。最佳条件仍为 C0，而最佳替代条件 C1 比 C0 低 10pp，未达到预注册的 +15pp 门槛。

结论：**IR_INTERFACE_NOT_MAJOR_BOTTLENECK**。C3 已把 schema validity 提升至 100%，却没有得到任何 semantic-success recovery；这把主要问题定位在 semantic family/operation/role/relation grounding，而不是 JSON 嵌套或 canonical serialization。该结论不涉及 downstream execution capability。

所有报告值均为 aggregate；不包含 task IDs、grids、test targets、oracle IR 或 raw response。
