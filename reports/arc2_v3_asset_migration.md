# ARC2 V3 Asset Migration

本报告分类现有资产；分类不修改任何历史 config、result、report、prediction hash 或 Git history。

## KEEP

- A3 deterministic raw-grid + bounded object/relation graph（V3 EvidenceBundle 直接复用）。
- connected components、bbox、颜色、面积、位置与 correspondence 计算。
- exact comparison、frozen split/cohort、prediction freeze 与 leakage controls。
- Kaggle orchestration、GPU mapping、checkpoint/telemetry 基础设施。
- 现有 deterministic grid primitives，作为 V3 executor 内部 backend 的候选实现。

## MIGRATE

- `llm.semantic_relation_normalizer` 的 deterministic relation feature 思路：迁入 V3 EvidenceBundle/CrossPairEvidence，而不是运行时 Semantic IR。
- 旧 ParameterSolver 的 finite candidate、direction/orientation、shape/color-source 解析：以 V3 `candidate_generator` + `joint_solver` 的显式 RuleSpec contract 迁移；旧实现目前是 legacy adapter，不被 V3 runtime import。
- CapabilityExecutor 的可执行 grid 操作：按 V3 canonical operation 逐项窄适配，隐藏在 `v3.execution` 内部。
- Hard verification：迁入 `v3.verification.HardVerifier`，并保持只能验证、不能 repair 的边界。

## LEGACY（保留科研历史，不进入 V3 runtime）

- ARCSEMANTICIRV1、semantic recognition 与其 input/model/interface ablation。
- Macro API comprehension、representation、compiler-aware benchmark 与 parameter repair 实验。
- DOWNSTREAM coverage forensics 与旧 results/reports/prediction artifacts。
- GRID_SKELETON_RECOGNITION_V1：实现和 commit 已保留；正式 run 被用户停止且不存在完整 frozen prediction artifact，状态为 `IMPLEMENTED_FORMAL_RUN_ABORTED`。不评分、不重跑。

## DEPRECATE_FROM_MAIN_PATH

- `LLM → Macro DSL → compiler → primitive program` 多层翻译。
- compiler-aware repair、second-LLM tool stitching、primitive repair loop。
- 将 capability registry 当作 LLM semantic label space。
- 任何通过 test target、oracle 或 execution coverage 反向决定 recognition 的 gate。

这些资产不删除；它们是 ADR-001 的实验依据。
