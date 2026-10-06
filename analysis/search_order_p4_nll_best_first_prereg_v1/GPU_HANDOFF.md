# P4 cumulative-NLL best-first — GPU handoff

## Status

CPU/offline research is complete through the GPU boundary.

Validated scientific-code commit:

`11956f80ca7a7cba242e830bfb86fd129f4ddf42`

Policy:

`CUMULATIVE_NLL_BEST_FIRST_V1`

Gate 0 is frozen PASS:

- controller compile/import: PASS
- search-order unit tests: 14/14 PASS
- shared-core contract tests: 4/4 PASS
- adapter/cohort guards: 3/3 PASS
- Gold loaded: false
- GPU generation: not started

Any branch HEAD used on RunPod must contain this commit as an ancestor. After this commit, only additive audit/handoff files are expected before GPU work unless explicitly reviewed.

## Scientific contract

The only scientific change relative to frozen P3 is logical retained-node pop order:

`(cumulative_nll ASC, insertion_order ASC, work_item_id ASC)`

Do not change:

- d24 historical adapter
- canonical AUG8
- cumulative-regret r=4 retention
- root-aware admission
- frontier floor = 1
- candidate cap = 32
- R1024
- EOS semantics
- physical scheduler/profile policy

No Gold may be loaded during generation.

## Frozen cohort

Use exactly:

`analysis/search_order_p3_lds_micro12_v1/MICRO12_COHORT.json`

SHA256:

`3852056112336e58559ceb0bb6f3171600ff36e0d6b02d1c9c03fc0da334443e`

Parent cohort:

`analysis/search_order_micro24_v1/COHORT.json`

Exact deterministic Gate-1 smoke outputs are:

- `1818057f:o0` — PROFILE_S
- `80a900e0:o0` — PROFILE_M
- `36a08778:o1` — PROFILE_L_LOW

## Validated 3090 runtime identity

Interpreter:

`/root/arc-runtime-3090-v5/env/3090-ampere-env-v2/bin/python`

Expected:

- torch 2.8.0+cu128
- CUDA 12.8
- transformers 4.55.4
- peft 0.17.1
- CUDA available = true
- RTX 3090

Recover the exact successful model/challenge/native-config/candidate-pool/coarse-policy paths from the previous P3/frontier-telemetry RunPod runtime. Do not substitute a merely similar model directory.

Frozen candidate-pool SHA256:

`a16e4194e7597e83e55c180896125638cfd595b108b4c935d5cd742058c90a13`

Canonical AUG8 SHA256:

`90585a8ac0462248b3be1a8a974851f03ad8167f686b466b286eed804d4151df`

Historical d24 adapters must audit as EXACT_HISTORICAL.

## Controller

`scripts/run_search_order_p4_nll_best_first_micro12_v1.py`

### RunPod Gate 0

Run the controller's `--mode gate0` once in the exact validated 3090 venv. This is CPU validation only.

Use a fresh P4 smoke root, for example:

`/root/arc-runtime-3090-v5/active_run/search_order_p4_nll_best_first_smoke_v1`

If it is already non-empty, do not overwrite it.

### Gate 1 — GPU smoke

Run `--mode smoke`.

Required PASS:

- 3/3 exact frozen smoke outputs
- 24/24 cells end as R1024 or legitimate terminal carry
- wall-censored cells = 0
- OOM fallback = 0
- duplicate work-item IDs = 0
- retention_changed = false
- completed candidates > 0
- generation hash verification = PASS
- Gold loaded = false

The completed-candidate condition is important: global cumulative NLL has a shallow-path bias. If the policy consumes R1024 without producing any candidate, classify this as `SCIENTIFIC_SEARCH_PATHOLOGY` and STOP. Do not weaken the policy.

Gate 1 produces:

- `GATE1_SMOKE_RESULT.json`
- `GATE1_SMOKE/GENERATION_FREEZE.json`
- `GATE1_SMOKE/GENERATION_HASH_VERIFICATION.json`

Do not access Gold after smoke.

### Gate 2 — frozen Micro12

Only if Gate 1 is PASS.

Use a separate fresh directory, e.g.:

`/root/arc-runtime-3090-v5/active_run/search_order_p4_nll_best_first_micro12_v1`

Run `--mode generate` and pass:

`--gate1-evidence <Gate1 smoke root>`

The controller fails closed if Gate-1 evidence is absent or invalid.

Expected generation:

- 12 frozen Micro12 outputs
- d24 × canonical AUG8 × R1024
- policy = CUMULATIVE_NLL_BEST_FIRST_V1
- target-blind
- Gold loaded = false

At completion require:

- 12/12 atomic output freezes
- generation hash verification PASS
- compact hash verification PASS
- `PRE_GOLD_FREEZE.json` status = FROZEN
- Gate1 status = PASS
- Gold loaded = false

Only after the pre-Gold freeze is committed may post-freeze Gold scoring begin.

## Why this is the next GPU experiment

Offline true-frontier evidence identifies cumulative NLL as the strongest robust search-order signal:

- Top1 = 32.91%
- Top2 = 48.43%
- Top5 = 80.72%
- MRR = 0.5204

Fair DFS and Classic LDS failed to rescue the Micro12 historical misses. Global/pairwise learned rankers generalized poorly. The preregistered conservative learned override also failed:

- trust-classifier LOOTO AUC = 0.579
- FIX = 3,651
- HARM = 6,311
- hybrid Top1 = 28.75% vs NLL baseline 32.91%

Therefore the cleanest remaining causal test is whether **live global cumulative-NLL best-first traversal** converts the strongest offline ranking signal into actual fixed-R1024 candidate recall.

Do not implement additional learned logic in this run.
