# ARC Parallel Diagnostic V2 — Synthesis

四条独立 track 都已完成；本文件只比较已冻结结果，没有新增推理、没有 combined-best rerun。所有 Recognition inference 均只看到 development TRAIN input/output pairs，随后才在本地隔离 oracle 层评分。

## Track A — Input representation

| Condition | Semantic success |
| --- | ---: |
| A0 raw grids | 0/30 (0.0%) |
| A1 deterministic features | 2/30 (6.7%) |
| A2 raw + current features (reused) | 4/30 (13.3%) |
| A3 raw + object/relation graph | 7/30 (23.3%) |

A3 最好，较 A2 为 +10.0pp，未达到预注册的 +15pp 门槛，且全部条件低于 25%。结论：**INPUT_REPRESENTATION_ALONE_INSUFFICIENT**。结构化关系信息显著改善 relational family coverage，但没有解决 iteration、conditional、composition semantic success。

## Track B — Model

Qwen3-8B M0 复用为 4/30（13.3%）。M1 没有已挂载的更强 generic 模型，记录为 unavailable，未下载模型。M2 是明确标记的 4B ARC-SFT；它完成 30 个请求但 30/30 无法解析为 `ARCSEMANTICIRV1`。因此结论为 **MODEL_CAPACITY_UNRESOLVED**，不能把 M2 的 specialized schema-incompatible 负结果解释为 generic capacity 证据。

## Track C — Semantic IR interface

| Condition | Schema valid | Semantic success |
| --- | ---: | ---: |
| C0 full JSON (reused) | 66.7% | 4/30 (13.3%) |
| C1 flat slots | 86.7% | 1/30 (3.3%) |
| C2 finite candidates | 70.0% | 0/30 (0.0%) |
| C3 hierarchical slots | 100.0% | 0/30 (0.0%) |

`INTERFACE_RECOVERY = 0/30`。C3 把格式有效率提升到 100%，却没有产生 semantic success，因此结论为 **IR_INTERFACE_NOT_MAJOR_BOTTLENECK**。

## Track D — Execution capability

五个 generic typed capability 已接入统一 V4 registry/executor，dispatch audit 为 5/5 expansion capabilities、无 unsupported V4 ID。representability 从 0/30 到 1/30（3.3%）：iteration 0/15、relational 0/8、conditional 0/4、composition 0/2、output construction 1/1。结论：**CAPABILITY_STILL_BLOCKED**。

## Final bottleneck ranking

1. **True semantic reasoning / field grounding**：最强 Recognition 仅 7/30；格式消歧没有 recovery。
2. **Capability coverage**：独立 execution audit 仍只有 1/30。
3. **Input representation**：A3 有真实 +10pp 增益，但未到决定性门槛。
4. **Model capacity**：因 M1 unavailable 仍无法判定。

已观察到的最强 Recognition 配置是 **A3 + Qwen3-8B + current full JSON IR**，7/30；这是 Track A 的观测结果，而非跨 track 调参或 rerun。

推荐唯一下一实验：**GRID_SEMANTIC_GROUNDING_V1**。它应固定 A3/Qwen3-8B/current IR，直接诊断 family、operation、role、relation 和 iteration field 的语义 grounding；本阶段不自动启动它。

新运行的 aggregate generation/audit time 为 2,636.9 秒（43.9 分钟；这是跨 worker 累计时间而非 wall-clock）。完整测试：**181 passed**。

公开 aggregate 不含 task IDs、grids、test targets、oracle IR 或 raw response；没有 held-out/public/challenge inference，也没有把 Recognition 与 execution 串联。
