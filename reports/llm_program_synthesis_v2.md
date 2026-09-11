# LLM Program Synthesis V2

Pilot is scored only after symbolic and direct predictions freeze. Wall-clock duration is reported separately from the sum of per-worker task runtimes.

| Condition | Tasks | Wall time | Macro schema | Macro type | Parameter resolution | Compile | Executable | Train consistent | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| symbolic_parameter_solver | 50 | 652.94 s | 0.402 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| direct_parameter_ablation | 50 | 637.70 s | 0.234 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Frozen deterministic baseline: 22 / 1000.
Pilot decision: **NO_GO**.
