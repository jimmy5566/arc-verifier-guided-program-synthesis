# Grid Recognition + End-to-End Diagnostic V1

## Hard-gate result

- Status: `BLOCKED_CAPABILITY_COVERAGE_BEFORE_FORMAL_A_B_C`.
- Scope: development split only; public evaluation, held-out, challenge-like and full-1000 inference were not used.
- Development pool after historical prompt-study exclusion: 435.
- Frozen DOWNSTREAM_V1 audit: 8,223,505 candidates generated, 3,739,652 compiler-valid, 4 train-consistent, 0 test-exact.
- Oracle prediction hash was frozen before the development solution file was opened: `91fba82af0da4c39133df1e50c5ee27573dc4e54d4872676626b0c45361d6353`.

## Conditions

| Condition | Status | Reason |
| --- | --- | --- |
| A Recognition Only | Not run | No 24/24 capability-covered cohort |
| B Oracle IR → Downstream | Not run | No test-exact oracle task exists in audited pool |
| C End-to-End | Not run | Formal prediction would confound recognition with known downstream coverage gap |

## Conclusion

The requested controlled diagnostic cannot validly start with frozen DOWNSTREAM_V1: the hard gate found zero test-exact oracle tasks, far below the required 24. This is evidence of a `DOWNSTREAM_CAPABILITY` bottleneck at the cohort-construction stage, not evidence about Qwen grid-recognition quality. No recognizer, GPU inference, prompt tuning, retry, or manual correction was run.

## Safety

- No raw ARC grids, solutions, oracle programs, model outputs, or test predictions are published.
- Oracle program details remain in an ignored local artifact; public artifacts contain aggregate counts and a prediction-freeze hash only.
