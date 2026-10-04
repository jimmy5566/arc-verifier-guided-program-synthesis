# L4 8-view root-bucket B4/B8 pre-launch audit

## Scope

This document freezes the infrastructure contract for
`L4_8VIEW_ROOT_BUCKET_B4_B8_SCALING_V1`.  It is target-blind hardware work:
there is no TTT, candidate generation, selector, gold data, or submission
path.

## Six-hour cancelled-run diagnosis

The cancelled B1--B16 run completed B1 through B12.  B16 missed its
600-second `MODEL_READY` deadline and the controller emitted `PHASE_ERROR` at
about 762 seconds.  Later B16 weight-loading output at about 22,673 seconds
proves that a torch loader survived the controller failure.

The notebook used unbounded `subprocess.run(...)`.  Once the controller
exited with failure, the notebook still waited for its child process tree to
terminate.  There was no outer process-group kill, so a surviving spawned
loader could retain the kernel/GPU allocation.  The old `_close_queue()` also
called `Queue.join_thread()` without a timeout; that is a separate unbounded
failure-cleanup hazard, though the available log cannot identify it as the
precise blocking stack frame.

This is `CONFIRMED_PROCESS_TREE_CLEANUP_GAP`, not evidence that a physical
forward needs six hours.

## New hard gates

| Gate | Value |
| --- | ---: |
| Notebook global limit | 1,800 seconds |
| Model-ready limit | 300 seconds |
| No-progress limit | 180 seconds |
| Per bucket/width limit | 600 seconds |
| Grace after SIGTERM | 10 seconds |

Every notebook phase launches with `Popen(..., start_new_session=True)`.  A
timeout records `TIME_GATE_FAILURE.json`, captures the last progress event and
`nvidia-smi`, sends SIGTERM to the entire child process group, waits at most
ten seconds, then sends SIGKILL if required.  It never falls back to an
unbounded wait.

GPU children do not start their own session.  They set Linux
`PR_SET_PDEATHSIG=SIGKILL` before importing torch, so controller death cannot
leave a model loader orphaned.  Failure cleanup drains needed records then uses
`cancel_join_thread()` and `close()`; it never calls an unbounded
`Queue.join_thread()`.

## Scientific freeze

The runner uses the production fixed 4+4 portfolio read from
`inference.d1_release_contract.PORTFOLIO`:

- TTT24: `flip_lr`, `flip_ud`, `transpose`, `anti_transpose`
- TTT48: `identity`, `rot90`, `flip_ud`, `anti_transpose`

All prompt views retain canonical pair order and `color_offset=0`.  The
benchmark is `BASE_MODEL_ONLY`, BF16, the existing Transformers-v5
DynamicCache/streaming split-and-adopt path, CPU root templates, two warmups
and twelve timed forwards per GPU.

Before timing, all eight actual roots are constructed and grouped by root
sequence length, request position, cache geometry, and cache key.  The run
requires exactly two compatibility buckets of four; otherwise it fails closed
with `8VIEW_BUCKET_ASSUMPTION_FAILED`.

For each real bucket, B4 measures its four views.  B8 is labelled
`COMPATIBLE_LANE_HARDWARE_SCALING`: it is two physical replicas of the same
four compatible views, not an assertion that one task's eight views can be
packed together.  The production-representative proxy remains
`median_latency(A_B4) + median_latency(B_B4)`.
