# Native public-reference A/B/C/D partial-cohort result

`frozen30` remains a diagnostic cohort.  Phase C was stopped by request.  Only a task with a valid atomic checkpoint containing all 32 augmentations and 128 raw generations entered the fixed partial cohort; no result below caused any task, ranker, or search setting to change.

## Frozen cohort

- Complete (25): `228f6490`, `2bcee788`, `31aa019c`, `3c9b0459`, `3f7978a0`, `40f6cd08`, `5207a7b5`, `60d73be6`, `62ab2642`, `6773b310`, `6d58a25d`, `6f8cd79b`, `72ca375d`, `996ec1f3`, `9f236235`, `a6953f00`, `bae5c565`, `bbb1b8b6`, `c8f0f002`, `d2abd087`, `dc2e9a9d`, `e3fe1151`, `e69241bd`, `e9ac8c9e`, `f8b3ba0a`.
- Incomplete/excluded (5): `8dae5dfc`, `b71a7747`, `cd3c21df`, `e48d4e1a`, `feca6190`.
- C persisted 3,200 raw branches and 1,411 unique candidates.  All 25 included checkpoints had 128 raw generations, zero invalid raw candidates, a valid candidate pool, and a matching checkpoint identity.

## Post-freeze score

| Method | N tasks | Top-1 | Two attempts | Any-of-K | Rescued vs A | Harmed vs A |
|---|---:|---:|---:|---:|---:|---:|
| A: original pool + original likelihood | 25 | 9 | 11 | 19 | 0 | 0 |
| B: A pool + public-reference view selection | 25 | 7 | 12 | 19 | 0 | 2 |
| C: 32 x 4 bounded beam-search pool + original likelihood | 25 | 8 | 9 | 22 | 0 | 1 |
| D: C pool + offline support/persisted-likelihood selection | 25 | 9 | 13 | 22 | 3 | 3 |

`Any-of-K` is diagnostic only.  Both attempts were selected before targets: attempt 2 is the next distinct output in the fixed ranking.

## Diagnosis

- A -> B: this particular view-based selection reduced Top-1 by two; it is not the dominant standalone improvement.
- A -> C: search increased candidate recall by three tasks (19 -> 22), but the original selector failed to exploit that extra recall (9 -> 8 Top-1).  Search is material for coverage.
- C -> D: offline equivalent-output support plus persisted likelihood increased Top-1 by one and two-attempt exact by four (9 -> 13).  Selection remains the bottleneck after search, and the two mechanisms are complementary.

The cautious Public-LB candidate is D with its predeclared two distinct attempts.  It ties A on Top-1, has the strongest two-attempt result, and preserves C's added candidate recall.  This is a frozen30 diagnostic recommendation, not a target-tuned claim.

## Compute saved

C used 11,810.77 generation seconds (12,290.60 task elapsed seconds across workers; 3.41 aggregate worker-hours).  Stopping before the remaining five tasks avoided 640 planned raw branches (16.7% of the 3,840-branch full budget).  Using the observed completed-task mean gives an estimated 2,458.12 worker-seconds, or 0.683 GPU-hours, avoided; this is an estimate rather than measured future use.

Artifacts are under `artifacts/public_reference_ablation_v32_partial/partial_cohort/`:

- `PARTIAL_COHORT_MANIFEST.json`
- `A_FROZEN.json`, `B_FROZEN.json`, `C_FROZEN.json`, `D_FROZEN.json`
- `PARTIAL_ABCD_SCORED.json`
