# Adaptive TTT Decision Log

## 2026-09-27 — Plan V1 frozen

**Decision:** authorize only the preregistered Step 1 depth x GEN_VIEW confidence-surface pilot described in [`ADAPTIVE_TTT_RESEARCH_PLAN_V1.md`](ADAPTIVE_TTT_RESEARCH_PLAN_V1.md).

**Scope:** 12 target-blind Eval60 development tasks, REF128 TTT, one continuous trajectory per task through depth 72, five checkpoint depths, and eight standard post-TTT generation views. Maximum 480 cells and 85 GPU minutes.

**Explicitly excluded:** real Eval60 test targets, full candidate generation, selector changes, augmentation-policy changes, multivariate routers, threshold tuning, Step 3, and production changes.

**Precondition:** this plan must be committed and pushed before GPU inference.

## Results

Pending Step 1 execution. No result, recommendation, or production implication is asserted by this plan-only entry.

## 2026-09-27 — Step 1 corrective CPU-only audit

**Execution status:** completed with a protocol deviation.  The original
preregistered workers produced **472/480** valid cells.  Worker 1 reached its
5,100-second deadline while processing `981571dc` (recorded elapsed time
5,187.43 seconds); its continuous original trajectory completed through depth
48.  A later worker reran that task and supplied the eight depth-72 cells from
a numerically divergent trajectory.  Those eight cells are explicitly marked
`POST_DEADLINE / MIXED_TRAJECTORY` and comprise the separate **480/480
EXTENDED_POSTHOC** sensitivity set.  They are not included in primary claims.

**H1 — confidence:** in the 472-cell primary surface, the best pooled pure
self-confidence signal was `sequence_logprob` (AUC 0.8524), while the best
macro within-task signal was `mean_token_entropy` (AUC 0.7002).  The best
pooled signal retained only 4/8 oracle-solvable tasks at Top-1, 4/8 at Top-2,
4/8 at Top-4, and 6/8 at Top-8.  Thus pooled association is not sufficient
evidence for a deployable confidence router.

**LOO competence (separate):** teacher-forced held-train-pair NLL is
target-blind with respect to Eval60 test outputs, but it is not pure Qwen
self-confidence.  Its primary pooled AUC was 0.9805 and macro within-task AUC
0.9857; its analytic per-task argmin-NLL replay retained 8/8 LOO-solvable
tasks.  This is an exploratory competence signal, not a selector or
production change.

**H2/H3 — depth and view coverage:** `b5ca7ac4` is depth-exclusive at 12 and
`cb2d8a2c` at 48, documenting heterogeneous useful depths.  Globally unique
view contributions occur for `71e489b6`, `7b3084d4`, `b5ca7ac4`, `cb2d8a2c`,
and `dfadab01`; two such contributions came from views with below-median mean
self-confidence.  The descriptive per-task oracle is joint=8,
depth-only=8, view-only=8, so this small surface does not establish a
material interaction advantage beyond separable per-task oracle choices.

**H4 / decision:** **REVISE_CONFIDENCE before large-scale GPU expansion.**
Preserve both primary and extended analyses, do not rewrite the original
frozen evidence, and do not promote the LOO competence replay to deployment
without a separately frozen validation plan.

Corrective artifacts:
`artifacts/adaptive_ttt_step1_depth_view_confidence_v1/corrective_audit_v1/`.
