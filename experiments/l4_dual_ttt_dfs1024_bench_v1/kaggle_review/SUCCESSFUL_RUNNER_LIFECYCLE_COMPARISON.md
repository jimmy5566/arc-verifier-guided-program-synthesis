# Startup lifecycle comparison

This document compares only process lifecycle mechanics. The scientific
workload in `run_eval60_reference_ttt_4gpu.py` is not copied into the
DFS1024 benchmark.

| Lifecycle property | Historical successful Eval60 runner | DFS1024 benchmark after control-plane patch |
| --- | --- | --- |
| Process context | `multiprocessing.get_context("spawn")` | Same |
| GPU binding | Worker `i` sets `CUDA_VISIBLE_DEVICES=i`, then binds local `cuda:0` | Same, with a persisted `CUDA_BOUND` milestone |
| Model ownership | One complete model per worker/GPU | Same |
| Ready handshake | Shared queue; parent consumes one ready message after each sequential spawn | Shared queue; four workers may initialize concurrently; parent validates and deduplicates by worker ID |
| Ready payload | Worker/GPU ID, GPU name, load time, VRAM | Same core evidence plus string/`None` UUID, compute capability, tokenizer/runtime provenance, and a synchronous pickle guard |
| Start barrier | Shared `Event`, set after four ready messages | Same, but set only after exact unique workers `{0,1,2,3}` validate |
| Failure evidence | Queue-only startup reporting | Each worker atomically owns status/failure files; disk evidence is authoritative if IPC is absent or late |
| Dead-worker handling | Cleanup in a `finally` block | Parent drains IPC, reads disk, freezes reports, releases the barrier, then terminate/join/kill cleans peers |
| Worker startup exit | Caught exceptions may return normally | Persisted startup failures explicitly exit nonzero |
| Cleanup bound | Join up to 90 seconds, then terminate | Poll <=2 s, queue grace <=1 s, terminate join <=5 s, kill join <=5 s; expected abort well below 30 s |

## Intentional differences

The historical runner starts and confirms workers sequentially. The frozen
benchmark already used concurrent four-worker startup, and changing that
would alter the measured infrastructure path. The patch retains concurrent
startup while making the handshake explicit, unique by worker ID,
pickle-safe, and recoverable from per-worker disk artifacts.

The benchmark retains its existing 1200-second worker-side barrier timeout for
healthy but slow startup. A detected failure no longer waits for that timeout:
the parent sets the barrier solely as part of abort cleanup and immediately
terminates peers. No TTT, augmentation, DFS, candidate, cohort, batching, or
rerun setting changes in this comparison.
