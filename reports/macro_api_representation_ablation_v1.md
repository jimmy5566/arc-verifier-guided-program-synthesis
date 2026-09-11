# Macro API Representation Ablation V1

## 冻结协议

- Qwen3-8B、local Transformers、BF16、4×NVIDIA L4、Internet Off。
- R1/R2/R3 各一次正式 pass；R4 复用 R2 原始输出，对每个无效输出最多一次 deterministic-feedback repair。
- 未加载 ARC grids、ARC solutions 或 task predictions。benchmark 仍为冻结的 60-case API-only 集合。
- raw / repaired model responses 仅位于忽略的 Kaggle/local checkpoint；本报告和公开 JSON 仅含 aggregate。

## 条件比较

| Condition | Schema | API | Type | Parameter | Compile | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 Typed Compact | 26.7% | 26.7% | 11.7% | 11.7% | 5.0% | 60 | 103328 |
| R2 Typed + Examples | 100.0% | 100.0% | 83.3% | 83.3% | 23.3% | 60 | 113462 |
| R3 Two-Stage | 20.0% | 20.0% | 20.0% | 20.0% | 0.0% | 115 | 179738 |
| R4 Verifier-Guided Repair | 85.0% | 85.0% | 83.3% | 83.3% | 23.3% | 106 | 209783 |

## R3 / R4 专项指标

- R3 skeleton type-valid: 91.7%; final compile-valid: 0.0%。
- R4 raw type / compile: 83.3% / 23.3%。
- R4 repair attempted / successful: 46 / 0；repair success rate: 0.0%。
- R4 final type / compile: 83.3% / 23.3%。

## 各类别最佳 compile-valid rate

- Single Macro: 40.0%
- Parameter Contract: 20.0%
- Two-Step Typed Chain: 0.0%
- Three-Step Composition: 33.3%

## 主要失败模式（最佳条件）

1. COMPILER_FAILURE/INELIGIBLE: 36
2. TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH: 10

## 结论

- BEST_API_PIPELINE: **R2_TYPED_EXAMPLES**。
- 状态: **API_STILL_BROKEN**；推荐下一实验: **GRAMMAR_CONSTRAINED_DECODING**。
- R0 baseline type / compile: 5.0% / 0.0%。
- Local tests: **114 passed**。
