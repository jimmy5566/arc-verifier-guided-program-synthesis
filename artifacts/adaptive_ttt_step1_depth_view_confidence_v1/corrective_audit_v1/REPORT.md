# Adaptive TTT Step 1 Corrective Audit V1

**Mode:** CPU-only, frozen artifacts only. Evaluation test targets were not opened.

## Provenance correction

- Primary preregistered dataset: **472/480** cells.
- Extended post-hoc sensitivity dataset: **480/480** cells.
- Confirmed deviation: worker1 exceeded the nominal 5,100 s wall slightly, then stopped during `981571dc`; the eight depth72 cells for that task came from a later restarted, numerically divergent trajectory and are excluded from primary analysis.
- No evaluation-test target was loaded; `loo_greedy_exact` is a held-out training-pair label.

## Corrected H1

- Best pooled primary self-confidence: `sequence_logprob` AUC = 0.8524.
- Best macro within-task primary self-confidence: `mean_token_entropy` AUC = 0.7002.
- Raw lower-is-better metrics remain raw in `REPORT.json`; only explicitly named oriented fields are sign-transformed.

## H4 / decision

The data are descriptive LOO pseudo-test evidence only.  They do not validate a deployment router. **Decision: REVISE_CONFIDENCE before large-scale GPU expansion.**

See `REPORT.json`, CSV tables, and `figures/` for complete primary/post-hoc separation and per-task evidence.
