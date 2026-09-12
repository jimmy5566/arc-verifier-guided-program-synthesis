# Grid Recognition Input Ablation V1

这是一个冻结的 30-task development-only Recognition 实验。A0、A1、A3 的预测在 Kaggle 4×L4 上完成并冻结后，才在本地隔离 oracle 层评分；A2 复用了 V1 已冻结预测。没有调用 Macro Registry、Compiler、Executor、HardVerifier、Search，也没有使用 test target。

| Condition | Schema valid | Family | Core operation | Role F1 | Relation F1 | Iteration semantic | Semantic success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 raw grids | 16.7% | 3.3% | 13.3% | 1.9% | 3.0% | 0.0% | 0/30 (0.0%) |
| A1 features only | 100.0% | 6.7% | 60.0% | 24.1% | 32.3% | 50.0% | 2/30 (6.7%) |
| A2 raw + current features (reused) | 66.7% | 16.7% | 36.7% | 9.8% | 13.3% | 16.7% | 4/30 (13.3%) |
| A3 raw + relation graph | 93.3% | 30.0% | 66.7% | 21.9% | 10.3% | 40.0% | 7/30 (23.3%) |

## Family-level semantic success

| Condition | Iteration 15 | Relational 8 | Conditional 4 | Composition 2 | Output 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 0 | 0 | 0 | 0 | 0 |
| A1 | 0 | 1 | 0 | 0 | 0 |
| A2 | 0 | 3 | 0 | 0 | 0 |
| A3 | 0 | 6 | 0 | 0 | 0 |

A3 是最佳独立输入条件。它相对 A2 的 semantic success 增益为 **+10.0pp**，低于预注册的 **+15pp** major-bottleneck 门槛；四个条件都低于 25%。因此结论是 **INPUT_REPRESENTATION_ALONE_INSUFFICIENT**。

关系图显著改善了 relational family success（A3 6/8，A2 3/8），也把 schema validity 从 66.7% 提升至 93.3%。但它没有解决 iteration、conditional、composition 的 semantic-success 缺口，且 A3 的 relation F1 并未超过 features-only 条件。这说明结构化表示有帮助，但当前主要限制仍包括更深的语义归纳与 IR field grounding；该结论不涉及执行能力。

运行时基础设施修复为 `PYTORCH_ALLOC_CONF=expandable_segments:True`。此前 V29 在无预测输出的 CUDA allocator fragmentation 阶段失败；V32 的 token preflight 上界为 A0 11,327、A1 5,887、A3 15,476，随后成功完成单次 90-request 冻结 run。

本报告不含 task IDs、grids、test targets、oracle IR 或 raw model response。
