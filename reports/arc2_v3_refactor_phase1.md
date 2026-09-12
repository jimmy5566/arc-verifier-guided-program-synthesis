# ARC2 V3 Refactor Phase 1

## 主路径

`Evidence Extraction → Rule Recognition → Parameter Inference → Executable RuleSpec → Deterministic Executor → HardVerifier`

- `v3.evidence`: 纯 deterministic EvidenceBundle；含 pair evidence、objects、changed cells、候选方向/距离/颜色及 A3 relation graph。
- `v3.evidence.cross_pair`: 对每个 train pair 的候选求交集，输出 CrossPairEvidence。
- `v3.recognition`: 仅定义 parameter-free RuleSkeleton recognizer protocol；预置 recognizer 只用于测试。
- `v3.parameters`: 不 import LLM；先 pairwise candidate generation，再 joint intersection，最后枚举有限 RuleSpec hypotheses。
- `v3.schema`: RuleSkeleton 只包含 canonical operation 与 typed slots；RuleSpec 只接受完整 resolved task-level parameters。
- `v3.execution`: 不 import Macro Compiler 或 LLM；直接执行 RuleSpec，并在每张 grid 运行 selector 以形成 instance binding。
- `v3.verification`: 全部 train pair exact match；仅返回 shape/color/location diagnostics，不会改变 RuleSpec。

## Phase 1 deterministic vertical slices

1. RECOLOR：`SELECT → RECOLOR`。
2. COPY/MOVE：`SELECT → MOVE`。
3. REPEAT：`SELECT → REPEAT`，跨三对 train pair 求得 `DIRECTION=(0,1)`、`STEP=2`。
4. RELATIONAL COPY：`SELECT → RELATIONAL_COPY`。
5. COMPOSITION：`SELECT → MOVE → RECOLOR → COMPOSE`。

所有 fixture 均经同一 V3 pipeline 和 HardVerifier exact validation。没有模型、test output、oracle、public/held-out/challenge 数据参与。

## Phase 1 边界

本阶段没有提交 Kaggle、没有运行 Qwen、没有运行正式 ARC benchmark、没有扩展 capability registry。下一阶段应只在 Phase 1 稳定后评估 `ARC2_V3_RULE_RECOGNITION_AND_PARAMETER_E2E_V1`。
