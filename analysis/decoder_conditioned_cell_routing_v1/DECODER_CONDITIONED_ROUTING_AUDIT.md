# Decoder-conditioned cell routing audit v1

## Scope

**MEASURED:** CPU-only retrospective audit of frozen Eval60 compact V2 Greedy and V5 artifacts. The primary surface is exactly depths `{12,24,48}` × views `{identity, flip_ud, transpose, anti_transpose}` = 12 cells per output. No model was loaded and no candidate, TTT, Greedy, TurboDFS, or Gold-path score was generated.

**INFERRED:** routing is evaluated as retrieval of frozen decoder-capable cells, not minimization of Gold NLL or train NLL.

**NOT ESTABLISHED:** hidden-test behavior or a deployment router.

## Decoder labels and coverage

| Coverage class | Outputs |
|---|---:|
| OBSERVED_POSITIVE | 33 |
| OBSERVED_NO_POSITIVE | 54 |
| CENSORED | 2 |

`OBSERVED_NO_POSITIVE` means no current decoder-positive cell exists in the observed 12-cell surface; it is **not** a router failure. Censored V5 records are preserved as `UNKNOWN`, never converted to false.

## Evidence availability

| Layer | Available current-Eval60 tasks | Status |
|---|---:|---|
| L1 SINGLE_HELDOUT | 12 | frozen held-out NLL evidence |
| L2 MULTIFOLD_LOO | 0 | finalized LOO artifacts were inventoried; each has zero task-id overlap with compact V2 Eval60 |
| L3 FULL_TRAIN_SAME_STATE | 0 | missing: same-adapter checkpoint hashes match current production in 0 task-depth states |

Three-way common cohort: **0 tasks**; three-way comparison is not established.

## Raw decoder-capability retrieval

Metrics are `Recall@1/2/4/6`; all are measured only on outputs with complete layer scores, complete label status for the target, and at least one positive cell. Small samples are labeled in the CSVs.

| Layer | Y_C Recall@1/2/4/6 | Y_R Recall@1/2/4/6 |
|---|---|---|
| L1 | 0.42857143/0.57142857/0.85714286/1.0 | 0.0/0.33333333/0.66666667/0.66666667 |
| L2 | NOT ESTABLISHED | NOT ESTABLISHED |
| L3 | NOT ESTABLISHED | NOT ESTABLISHED |

## Decoder-calibrated out-of-task retrieval

Calibration is a single-feature regularized logistic model trained only on the layer's frozen train-side scalar, with leave-one-task-out splits. It is development analysis; no random-cell splits and no in-sample result are reported as evidence.

| Layer | calibrated Y_C Recall@1/2/4/6 |
|---|---|
| L1 | 0.42857143/0.57142857/0.85714286/1.0 |
| L2 | NOT ESTABLISHED |
| L3 | NOT ESTABLISHED |

## DFS-only rescue evidence

Known V5-only rescue output case studies are in `dfs_rescue_case_studies.csv`. A missing rank is reported as unavailable rather than inferred. The raw Y_R conclusion is **INSUFFICIENT**.

## Decision

**MEASURED:** observed-surface coverage bottleneck classification is **HIGH** using the share of observed-no-positive outputs.

**INFERRED:** best supported layer is **NOT_ESTABLISHED**; best supported K is **NOT_ESTABLISHED**. This is deliberately `NOT_ESTABLISHED` unless a shared comparison cohort contains at least ten positive Y_C outputs.

**NOT ESTABLISHED:** L3 ranking, a three-way winner, an image/surface model, or safely reducing the production surface to 6/4/2 cells. `IMAGE_MODEL_READINESS = PREMATURE`.
