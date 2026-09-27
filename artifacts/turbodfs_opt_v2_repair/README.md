# TURBODFS OPT V2 rapid structural repair

## Scope

This target-blind repair reused six immutable calibration adapters from the
failed V1 run. It did not rerun TTT, inspect evaluation solutions, modify V1
artifacts, or optimize candidate correctness.

## V1 anatomy

The CPU-only trace audit found 19 zero-complete V1 cells: 13 terminated at the
V1-only `max_batch_forward_passes=256` cap and 6 exhausted the frontier after
the frozen cumulative-NLL pruning gate. No V1 trace supplied evidence for an
EOS detector, parser, serialization, deduplication, cache, or lane-state
failure.

The full generated anatomy CSV is retained with its target-blind run at
`/workspace/arc2/active_runs/eval60_adaptive_inference_joint_v2_repair/repair_artifacts/turbodfs_v1_failure_anatomy.csv`
with SHA256 `af63127540f2969e7e8065a2461b7be075959814b0a7934a69999930cd66f1f8`.

## V2 repair and result

V2 removed the non-public 256-forward cap and restored the public notebook's
540-second time-only stop. It retained the NLL threshold, native vocabulary,
EOS, maximum length, TTT state, views and parser.

The deterministic six-cell micro calibration returned 4/6 complete valid
candidate cells, below the preregistered 5/6 gate. The two remaining failures
were `search_exhausted` without a branch cap, so the budget-only V2B retry is
not authorized. The 24-cell calibration and the authoritative Eval60 run were
not launched.

This is structural decoder evidence only, not an accuracy result.
