# Macro API Comprehension Benchmark V1

## Protocol

- Model: Qwen3-8B Competition Model through local offline Transformers (BF16).
- Hardware: four independently loaded NVIDIA L4 workers with sequential artifact warm-up, staggered readiness, then a shared task queue.
- Formal runs: exactly one frozen 60-case run; temperature 0, thinking disabled, candidate budget 1, max new tokens 256.
- ARC data used: **No**. ARC solutions used: **No**. This benchmark does not execute grids or score task predictions.
- Canonical validation before the run: schema/type/compile **60/60**; benchmark hash `90aba782703a8b1d2a4e97c8cde81a5e9e04aa0ca673ec882c70e42e0245ff00`.
- Raw completions and parsed programs remain in ignored Kaggle/local checkpoint artifacts. This report and the public JSON contain aggregate metrics only.

## Overall funnel

`60 → 60 → 60 → 12 → 12 → 3 → 3 → 3 → 0`

| Metric | Rate |
| --- | ---: |
| JSON parseable | 100.0% |
| Schema valid | 20.0% |
| Macro API valid | 20.0% |
| Type valid | 5.0% |
| Parameter valid | 5.0% |
| Compile valid | 0.0% |

## Category metrics

| Category | Cases | Schema valid | Type valid | Compile valid |
| --- | ---: | ---: | ---: | ---: |
| Single Macro | 15 | 60.0% | 0.0% | 0.0% |
| Parameter Contract | 15 | 20.0% | 20.0% | 0.0% |
| Two-Step Typed Chain | 15 | 0.0% | 0.0% | 0.0% |
| Three-Step Composition | 15 | 0.0% | 0.0% | 0.0% |

## Failure taxonomy

1. PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER: 46
2. TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH: 9
3. COMPILER_FAILURE/INELIGIBLE: 3
4. MACRO_API_FAILURE/INVALID_ARGUMENT_NAME: 2

V2 API failure pattern reproduced: **YES**.

## Pre-registered decision

**NO_GO_API** — recommended next experiment: **API_REPRESENTATION_ABLATION**.

The frozen API-only condition did not meet the pre-registered reliability gate. ARC reasoning remains confounded with Macro DSL use, so no further ARC pilot should be inferred from this result.

## Research questions

- **Q1 Macro names:** 100.0%
- **Q2 Argument contracts:** 20.0%
- **Q3 Parameter wrappers:** 5.0%
- **Q4 Input/output types:** 5.0%
- **Q5 Legal two-step chains:** 0.0%
- **Q6 Legal three-step chains:** 0.0%
- **Q7 V2 type-valid=0 primarily API teaching:** YES
- **Q8 ARC reasoning can now be meaningfully evaluated:** NO

## Verification

- Local test suite after implementation: **105 passed**.
- Registry hash: `d7309e5f8c4f255c90a2c9866254c85710c627854c0438a784e9598dfa490951`
- Schema hash: `48edf7f64ca095ce75ebf640a542366e10fc784b1a81ad82f6f5837b0ea658fa`
- Frozen config hash: `a8cd630eb786f5da14d1fab9079279949817860cfea2f4d7510a2cb02e541d79`
