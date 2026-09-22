# SMOKE12: TTT Cost / Accuracy Study

Status: planned only. Do not launch S1 or S2 without an exact GPU approval record.

## Conditions

| ID | Condition | Assignment | Execution |
| --- | --- | --- | --- |
| S0 | Existing frozen strong-TTT 24-step greedy baseline | Reuse only | No rerun |
| S1 | Strong-TTT, 24 steps, Beam2 | Experiment Runner A | Pending approval |
| S2 | Strong-TTT, 48 steps, greedy | Experiment Runner B | Pending approval |
| S3 | Base plus TTT candidate union | Auditor | CPU-only |

## Metrics

Primary metric: Any-of-K.

Cost metrics:

- Wall-clock time.
- GPU seconds.
- Seconds per task.
- Peak VRAM.
- GPU seconds per new recovery.
- Projected 240-task runtime.

The study remains a SMOKE-sized decision aid. It does not automatically expand to CONFIRM or VALIDATE.
