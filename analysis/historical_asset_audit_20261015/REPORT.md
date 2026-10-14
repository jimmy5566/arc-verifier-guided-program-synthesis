# ARC2 historical asset audit — 2026-10-15

## Scope and decision

This was a read-only audit of historical TTT, augmentation, TurboDFS, selector,
V7, and competition-release evidence on `codex/connected-components-e1`.
No model was loaded, no candidate was generated, no GPU work was launched, no
Gold/dGold target was opened, and no release or submission was created.

The Phase3 D24/D48 archives, their base model, the 180 task-specific adapters,
and V7 are available.  D24 is CPU-replay ready and has same-contract numerical
reproducibility evidence.  D48 archive bytes are ready, but the historical D1
TTT48 model-state parity question remains unresolved.  TurboDFS V5 is partial
and its live state database is malformed, so it is not a complete scoring
asset.  Existing selector results are exposed-development evidence and must not
be treated as independent confirmation.

## Immutable Phase3 D24/D48 archives

The live RunPod archives were fully traversed with the repository validators.

| Archive | Model state | Outputs / tasks | Cells | Candidate hashes | Current validation | GPU wall | Fallbacks | Compact ledger SHA256 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| D24 AUG8 R1024 | base `qwen3_4b_grids15_sft139` + depth-24 task adapters | 89 / 60 | 712 | 1,024 | PASS; 0 missing, mismatched, malformed, or duplicate output IDs | 31,561.630 s (8.77 h) | 1 | `9b37c43b030c0fc66a36eaabeddf86b0d7f584bfb7993206abc30976780c92d8` |
| D48 AUG8 R1024 | same base + depth-48 task adapters | 89 / 60 | 712 | 1,021 | PASS; 0 missing, mismatched, malformed, or duplicate output IDs | 30,353.351 s (8.43 h) | 0 | `6bb97ee885bc57128f831bbdb584d7fab5c8caf484ff139435325a27e3a729d0` |

Both source audits passed and both manifests record `gold_loaded=false`.  The
CPU parser/replay surface is ready: 31 focused tests covering D24/D48 archive
contracts, TTT24 parity repair, TurboDFS V5, selector D1, and D1 replay passed
in 4.38 seconds in the final audit run.  This proves current CPU parsing/replay compatibility; it is
not new GPU numerical-parity evidence.

The Phase3 post-freeze score on 60 tasks / 89 outputs found candidate coverage
(Any-of-K) on 19 D24 tasks, 18 D48 tasks, and 22 union tasks.  Output coverage
was 31 D24, 30 D48, and 35 union outputs.  NLL Top1 was 16/89 for both D24 and
D48 and 14/89 for the union ranker.  NLL Top2 was 23/89 (13 tasks), 21/89
(11 tasks), and 25/89 (15 tasks), respectively.  Support-NLL Top1 was 19/89,
15/89, and 19/89; Support-NLL Top2 was 25/89 (14 tasks), 23/89 (13 tasks), and
24/89 (14 tasks).  Top2 was descriptive and not a preregistered Phase3 metric.

## Live model and raw-asset inventory

| Asset | Live finding | Classification |
| --- | --- | --- |
| `qwen3_4b_grids15_sft139` | 7,267,274,168 bytes; pinned identity `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`; all 13 manifest files freshly rehashed; manifest SHA256 `9e995b611936791884063b69ef85cba12d89c53dd1c6cebe33ac155145c9aee7` | VERIFIED_READY |
| 180 task TTT adapters | 60 each at depths 12/24/48; 190,295,612,640 total bytes; 180/180 paths and sizes match; immutable manifest contains full hashes; 15 adapters were freshly byte-rehashed with no mismatch | VERIFIED_READY, with current full-byte rehash sampled rather than repeated across all 177.2 GiB |
| V7 adapter | `adapter_config.json` SHA256 `faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69`; `adapter_model.safetensors` SHA256 `1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3`; both freshly rehashed and match the repository manifest | VERIFIED_READY |
| TurboDFS V5 raw run | planned 1,068 cells but only 196 DONE / 872 PENDING; 44 parquet shards, 89,903,664 bytes, ledger SHA256 `2c33caea0475c7b2c9d902d836ece93fd3bc7c17934010eac83b69a8ba8790f3`; state DB SHA256 `cdeee7a42de3fbba11fafda943be609e1ce6c49cb7c36dbb7d1262a1ccfe33aa` and currently reports `database disk image is malformed` | NEEDS_REPAIR; partial/nonblind evidence only |
| TurboDFS V4 reference/calibration | historical calibration/reference material; not a complete current experiment | HISTORICAL_ONLY |

The 180-adapter qualification deliberately combines the prior immutable full
hash attestation with current 180/180 existence/size checks and 15 fresh
full-byte samples.  Rehashing all 177.2 GiB again was stopped as low-information
I/O, so the report does not claim a fresh full rehash of all adapters.

## Cohort-separated historical metrics

Metrics are normalized in `METRICS_BY_COHORT.csv`.  Important separations are:

- Phase3 Eval60 D24/D48 uses the task-specific TTT adapters, not V7.
- V7's 49/60 native-Greedy Top1 and 51/60 complete-output Top2 are from a
  reused development cohort; LoRA-disabled was 44/60 and 45/60.  This is not
  independent ARC generalization and is not comparable to Phase3's candidate
  archive coverage.
- Frozen60 cached-pool results are 30/60 Any-of-K, 16/60 Top1, and 21/60 Top2.
  Frozen30 is task-disjoint but shares project/model context; it is not an
  independent statistical replication.
- The clean five-task TTT validation improved Any-of-K from 1/5 to 2/5 and
  Top1 from 0/5 to 1/5 at 949.18 s baseline versus 1,153.81 s TTT.  This small
  result established one target-blind candidate conversion, not a broad gain.
- The partial frozen30 A/B/C/D study had A 19/25 Any-of-K, 9/25 Top1, 11/25
  two-attempt; C 22/25, 8/25, 9/25; and D 22/25, 9/25, 13/25.  Five tasks were
  excluded, so it is a partial diagnostic.
- Phase3 selector S2 improved exposed-cohort Top2 from 25/35 to 27/35 while
  Top1 stayed 19/35.  Later source hedges, robustness features, and train
  constraints did not establish a robust additional selection signal.
- The D1 historical replay is exact at 20 Top1, 28 Top2, and 30 pool-oracle
  hits over 89 attempts.  It is replay evidence, not a live release score.

## TTT24/TTT48 parity and D1 release

The original TTT24 historical-pilot comparator diverged numerically at step
507 despite the same node sequence; the audit classified it as model-call
history/numerical divergence and retired that comparator as confounded.  The
later same-contract reruns are stronger evidence: 24/24 S/M cells and 8/8 L
cells were strict semantic-and-numeric matches.  The consolidated TTT24 core is
`CORE_CLASS_READY` with 32 exact cells and no Gold access.  Per-forward cache
owner traces remain absent in those reruns, so that telemetry is unverified.

TTT48 remains a D1 blocker: the old release cross-score reported 50 task-anchor
mismatches and `FAILED_NUMERICAL_PARITY`.  Ready D48 candidate bytes do not
resolve that model-state question.

`release/TTT24_TTT48_4PLUS4_D1_BASELINE_V1/RELEASE_STATUS.json` is a
pre-submission snapshot and must not be used as current status.  The later
immutable release record proves that Kaggle notebook Version 1 was created and
submission `56508996` was submitted on 2026-09-24; its last durable event was
`PENDING` with no public score.  Current Kaggle status is UNAVAILABLE because
neither the local nor RunPod environment has a usable read-only Kaggle CLI
credential.  No credential was requested or moved.

## Disproved, exhausted, or non-repeatable directions

- Do not repeat exposed Eval60 selector tuning: S2's +2 Top2 is unconfirmed;
  source hedge S3A was net -4 versus S2 and S3B was net 0; robustness and
  train-constraint signals were not supported.
- Do not repeat the historical sparse TTT24 branch variants as-is: the small
  frozen smoke produced no Any-of-K or Top2 gain.
- Do not claim exact reuse of the historical AUG8 hard28 pool: 0/112 omitted
  cells had sufficient adapter/tokenizer/recipe provenance for exact reuse.
- Do not score the incomplete V5 run as a full TurboDFS result.  It is partial,
  nonblind recovery evidence and its state database needs repair.
- Do not rerun the original confounded TTT24 parity comparator; use the later
  same-contract result.  TTT48 parity, however, remains genuinely unresolved.

## Highest-return next score experiment

`INDEPENDENT_FROZEN_POOL_SELECTOR_CONFIRMATION_V1` is the recommended next
score experiment, but it was not started here.  Freeze one simple selector
choice before target access (historical B versus S2), evaluate it once on a
fresh task-disjoint target-blind candidate pool made with one pinned model
identity, and report paired Top1/Top2 at output and task level plus Any-of-K.
This directly tests whether the only promising exposed selector delta
(25/35 to 27/35 Top2) transfers.  It has higher expected information value
than more Eval60 feature tuning, another sparse TTT branch, or attempting to
interpret the incomplete V5 run.  If no genuinely unexposed frozen pool is
available, the experiment remains NOT_RUN rather than reusing exposed targets.

## Evidence pointers

- `analysis/eval60_phase3_d24_d48_r1024_score_v1/SUMMARY.json`
- `analysis/phase3b_eval60_d48_aug8_r1024_v1/REPORT.md`
- `analysis/ttt24_aug8_r1024_parity_repair_v1/POST_FREEZE_RECONCILIATION.json`
- `analysis/ttt24_aug8_r1024_reproducibility_l_single_v1/CORE_DECISION.json`
- `analysis/eval60_phase3_r1024_exact_b_selector_v8/SUMMARY.json`
- `analysis/eval60_phase3_selector_v2_forensic_v1/SUMMARY.json`
- `analysis/eval60_phase3_selector_v21_source_complementarity_v1/SUMMARY.json`
- `analysis/eval60_phase3_selector_v22_robustness_forensic_v1/SUMMARY.json`
- `analysis/eval60_phase3_selector_v23_train_constraint_forensic_v1/SUMMARY.json`
- `analysis/augmentation_candidate_screening_v1/DECISION.json`
- `analysis/historical_eval60_aug8_reuse_v1/DECISION.json`
- `release/TTT24_TTT48_4PLUS4_D1_BASELINE_V1/RELEASE_STATUS.json`
- `governance/releases/d1-submitted-v1/release.lock.json`
- `governance/releases/d1-submitted-v1/submission_events.jsonl`
- `experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json`

