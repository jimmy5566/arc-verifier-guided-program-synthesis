# ADR-001：采用 V3 RuleSpec 主路径

状态：Accepted（Phase 1）  
日期：2026-09-12

## 决策

ARC2 的新主路径固定为：

`Evidence Extraction → Rule Recognition → Parameter Inference → Executable RuleSpec → Deterministic Executor → HardVerifier`。

五层各自只有一个职责：Evidence Extraction 计算可确定事实；Rule Recognition 只识别最小 operation skeleton；Parameter Inference 只对所有 train pair 的有限候选进行交集、约束传播与 joint validation；RuleSpec 绑定 task-level rule parameters；Executor 直接运行 RuleSpec 并在每张 grid 上计算 instance bindings；HardVerifier 只报告 exact-match 结果与诊断，绝不修复规则。

## 背景与证据

旧路径包含 `Grid → Semantic IR → Macro DSL → Compiler → Primitive Program → Executor`。它并非任意堆积：每个局部实验都针对当时发现的 failure 加入约束或接口。不过累积后的多次语义翻译带来了局部优化，未带来相应的端到端收益。

- Macro API / compiler-aware 实验显示 API grounding 与编译接口是瓶颈。
- parameter grounding 实验表明参数值与规律结构应解耦。
- A3 raw-grid + object/relation graph 提升了识别输入质量，但没有使完整 semantic IR 稳定。
- GRID_SEMANTIC_RECOGNITION_V1 为 4/30 semantic success；A3 为 7/30；full exact 为 0。
- frozen DOWNSTREAM capability audit 只有 1/30；direct grid solve 为 1/30。

因此 V3 不把 capability coverage 当作 recognition gate，也不让第二个 LLM 决定工具调用。

## 约束与后果

- RuleSkeleton 使用小的 canonical operation vocabulary 与 typed slots；它不含颜色、方向、距离、坐标或对象编号等具体值。
- Rule parameters 跨全部 train pairs 固定；instance bindings 在 executor 内每张 grid 独立计算。
- retained primitive/capability 实现只是 execution backend，不再是 LLM-facing reasoning API。
- V3 Phase 1 与旧 Macro、Compiler、Semantic IR、repair 链并存（strangler refactor），不删除历史科研资产。
- 后续 top-K skeleton 由 `RuleRecognizer` protocol 支持；Phase 1 不运行任何新的 LLM inference。
