# Masked Batch2 root-cause performance audit

## Evidence scope

CPU analysis of frozen A1.1 raw evidence plus a separately labelled contaminated
performance-only 97d7923e microprofile. The microprofile did not access Gold.
Frozen A1.1 manifest SHA256: `ce6afb4fc5f33901fa7bbe0813d7df7cae7f24a12f5a85960bf2298d79a1673b`.

## Occupancy

- Physical forwards: 12907
- Active one-lane forwards: 12160 (94.2124%)
- Active two-lane forwards: 747 (5.7876%)
- Effective batch: 1.057565662
- Effective lane utilization: 52.8783%; wasted lane fraction: 47.1217%

The physical tensor width remains two for active-one calls: `PHYSICAL_BATCH2_WITH_PAD`.

## Root cause

The corrected engine no longer lets an inactive pad/cache lane enter the decoder.
The pre-mask apparent occupancy near two was therefore **INACTIVE_LANE_REACTIVATION**,
not valid parallel search. The frozen masks show most calls have one live logical
state. Group 1 is the strongest scheduler evidence: both lanes end at identical
node caps, but 7,120 physical calls are singly active (3,560 observed from each lane).
That is recursive lockstep exposing work at different stack levels, not merely a
permanently finished companion cell. Group 0 additionally has real early terminal
asymmetry (candidate cap versus node cap).

## Timing microprofile

| metric | scalar B1 | Batch2 active=1 | Batch2 active=2 |
|---|---:|---:|---:|
| mean CUDA forward ms | 58.2492 | 64.9881 | 66.1001 |
| count | 5577 | 5040 | 210 |

Batch2 model-forward fraction: 97.08%; residual CPU/orchestration
fraction: 2.92%. Detailed timers are in `batch2_time_breakdown.csv`.
`KV_PACK_OR_SELECT_SECONDS` is zero on this executor: it passes an already
batch-shaped cache directly; it does not compact/select cache rows.

## Ceiling and architecture implication

Even under an impossible optimistic assumption—two-wide forward latency equal to
scalar and zero overhead—the fixed-pair ceiling is the measured effective batch,
only **1.0576x**. The actual microprofile model-only comparison
is **0.9470x**, because an active-one Batch2 forward is slower
than a scalar forward. Therefore >1.15x cannot be obtained while occupancy
remains at this level. A dynamic ready pool could raise occupancy by
batching independent ready states, but it would be a **new decoder execution
architecture**, and it would still retain the known BF16 batch numerical path
divergence. It is not scalar-equivalent engineering.

## Gold / science status

Post-freeze scoring already established that the two known d59 canaries survived
in Batch2 (2/2). This audit neither rescored Gold nor changes the historical 33/89.
