# Stage-2A leakage and triviality audit

## Separation

`episodes()` returns telemetry, observable state, and hidden source separately.
The attribution model is fit only as `model.fit(xtr, ytr)`: the telemetry matrix
has the 14 documented observable columns and has no source/failure column. The
online predicted updater receives model probabilities only. It never receives
the hidden source.

The hidden source appears only in two deliberately non-online places:

1. outcome accounting for the global/state-naive baselines and final metrics;
2. the labelled `ORACLE_STATE` upper-bound comparator.

The latter is explicitly labelled oracle and is not an implementable method.

## Observable state is not a source shortcut

States are deliberately observable because the question is state-conditional
reliability. A state-only random-forest audit on held-out episodes reached
macro-F1 0.256 and selector-failure F1 0.415, versus 0.705 and 0.840 for the
full noisy-telemetry model. Thus state prevalence is informative but not a
deterministic hidden-source label.

## Target and environment boundaries

This directory contains a seeded synthetic environment only. It does not read
ARC grids, ARC solutions, frozen cohorts, production candidates, network
resources, models, CUDA, or Kaggle inputs. The degraded-attribution ablation is
constructed only from calibration-source prevalence and learned telemetry
probabilities; it never uses held-out source labels for updating.
