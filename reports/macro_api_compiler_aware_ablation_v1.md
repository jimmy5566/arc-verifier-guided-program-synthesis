# Macro API Compiler-Aware Ablation V1

## 冻结协议

- C0 仅引用冻结的 R2 aggregate，未重新生成；C1/C2/C3 各一轮正式 pass。
- Qwen3-8B、local Transformers、BF16、4×NVIDIA L4、Internet Off、temperature=0、thinking=false、candidate budget=1。
- 未加载 ARC grids、ARC solutions 或 task predictions。raw responses 仅在忽略的 checkpoint 中。

## 条件比较

| Condition | Schema | Type | Parameter | Compile | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 R2 Typed + Examples（冻结历史） | 100.0% | 83.3% | 83.3% | 23.3% | 60 | 113462 |
| C1 Compiler-Aware Catalogue | 1.7% | 0.0% | 0.0% | 0.0% | 60 | 75576 |
| C2 Compiler-Valid Skeleton | 0.0% | 0.0% | 0.0% | 0.0% | 120 | 94006 |
| C3 Structured Compiler-Constrained | 100.0% | 100.0% | 100.0% | 100.0% | 60 | 181487 |

## C2 / C3 结构指标

- C2 skeleton JSON / selection-valid / canonical-sequence-match: 60 / 60 / 32（每项分母 60）。
- C3 structured JSON / skeleton-valid / parameter-constraint-valid / canonical-sequence-match: 60 / 60 / 60 / 30（每项分母 60）。

## 各类别最佳 compile-valid rate

- SINGLE_MACRO: 100.0% (C3_STRUCTURED_COMPILER_CONSTRAINED)
- PARAMETER_CONTRACT: 100.0% (C3_STRUCTURED_COMPILER_CONSTRAINED)
- TWO_STEP_TYPED_CHAIN: 100.0% (C3_STRUCTURED_COMPILER_CONSTRAINED)
- THREE_STEP_COMPOSITION: 100.0% (C3_STRUCTURED_COMPILER_CONSTRAINED)

## 最佳条件 failure modes

- 无

## 限制与解释

- C1 的主要失败为 `PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER`；C2 的主要失败为参数 wrapper 失配及结构约束拒绝。
- C3 的 compiler-valid 是由预先冻结的全局 compiler-valid skeleton/slot grammar 保证，而不是从 canonical answer 选择或自动修补得到。
- C3 skeleton canonical-sequence-match 为 30/60；因此 100% compile-valid 只证明 API/编译接口可达，**不**证明 transformation intent 或 ARC reasoning 已正确。

## 结论

- BEST_API_PIPELINE: **C3_STRUCTURED_COMPILER_CONSTRAINED**；状态: **API_FIXED**。
- C0 R2 type / compile: 83.3% / 23.3%；最佳 C1--C3 type / compile: 100.0% / 100.0%。
- 推荐下一实验: **ORACLE_LADDER**。
- Local tests: **123 passed**。
