# G2 TTT0 hard-miss attribution

NONBLIND_MECHANISM_DIAGNOSTIC

Gold scoring: G0.5-validated INCREMENTAL_KV_REPLAY only.
TTT0 Greedy Gold labels were attached only after its candidate cells were frozen.

Outputs: 28; TTT0 Gold cells: 112; TTT0 Greedy cells: 112
TTT0 strict-searchable outputs: 0
TTT0 V5-searchable outputs: 0
TTT0 Greedy exact outputs: 0
New rescue IDs: []
Current development union: 33/89; new nonblind development union: 33/89

## Depth win counts (ties count for each co-winner)
- NLL/token: TTT0=5, TTT12=12, TTT24=8, TTT48=3; tied outputs=0
- prune fraction: TTT0=14, TTT12=11, TTT24=19, TTT48=17; tied outputs=16
- mean legal rank: TTT0=8, TTT12=14, TTT24=13, TTT48=11; tied outputs=11

## Attribution classes
- NO_CLEAR_TTT_EFFECT: 15
- TTT_DEPTH_SENSITIVE: 3
- TTT_HARMFUL_SOFT: 1
- TTT_HELPFUL_SOFT: 9

## Required answers
1. TTT0 better on hard misses: 1 harmful strong/soft outputs; exact TTT0 rescues=0.
2. Deeper TTT systematic worsening: see per-output comparison; no aggregate causal claim is made beyond these nonblind 28 outputs.
3. TTT12 better than TTT0: inspect metric win counts and per-output classes; TTT12 wins are not collapsed into a fake composite metric.
4. TTT0 V5-searchable outputs: 0.
5. TTT0 Greedy direct new exact solves: 0.
6. Dominant next bottleneck: AUGMENTATION_COVERAGE_SUSPECT; recommended next experiment: AUGMENTATION_COVERAGE_TEST.
