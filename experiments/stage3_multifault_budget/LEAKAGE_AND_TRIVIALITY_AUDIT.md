# Stage-3 leakage and non-triviality audit

## Hidden multi-label separation

`episodes()` creates the latent four-bit fault vector separately from the
14-column observable telemetry matrix. The full attribution router consumes
only four independently predicted marginal probabilities. It never receives the
fault vector or the mutually exclusive category index. Faults are accessed only
for simulation outcome sampling, evaluation, and the explicit non-deployable
oracle comparator.

The four component classifiers and the binary failure-risk model train only on
the 10,000 calibration episodes. The 2,500 validation episodes are created but
not used to tune actions, thresholds, costs, or any test decision. The OOD
dual-fault model is not retrained.

## Tests against simple uncertainty

The comparison includes raw uncertainty-only routing, generic binary
failure-risk routing, state-only marginal responsibilities, and a fixed
permutation of full probability vectors. Shuffling retains the full model's
action-frequency marginal distribution but removes episode-level alignment.

On ID budget-1 MEDIUM, full attribution utility is .519, while State-Only is
.406, Uncertainty-Only .336, Failure-Risk-Only .349, and shuffled attribution
.362. On budget-2 the corresponding figures are .589, .446, .337, .446, and
.374. The contribution therefore depends on component-specific probability
alignment rather than generic intervention rate or state prevalence.

## Scope limitation

The environment permits at most two faults and uses fixed, independent partial
repair probabilities. Full attribution has informative but imperfect synthetic
telemetry. Stage-3 establishes decision value in a bounded ambiguous causal
setting; it does not establish real-agent performance, causal identification, or
robustness to triple/feedback-loop failures.
