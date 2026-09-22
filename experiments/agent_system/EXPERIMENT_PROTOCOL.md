# ARC2 historical optional multi-agent experiment protocol

> **Optional / legacy mode.** Normal ARC2 work uses the single-agent rules in
> the repository-root `AGENTS.md`. This document preserves a prior
> orchestrator/runner/auditor process for cases where explicit parallel
> isolation is worth its coordination cost; it is not a prerequisite for a
> routine smoke experiment or CPU development.

When this legacy mode is explicitly selected, its studies follow this
non-automatic sequence:

`Protocol Freeze -> CPU Precheck -> Pre-run Audit -> GPU Approval -> GPU Run -> Artifact Freeze -> Scoring -> Post-run Audit -> Orchestrator Decision`

No experiment may automatically expand to the next stage.

## Stages

| Stage | Tasks | Typical runtime |
| --- | ---: | --- |
| SMOKE | 6-12 | 10-40 minutes |
| CONFIRM | 20-30 | 30-90 minutes |
| VALIDATE | 40-60 | Target below 2 hours |

The protocol freeze records the research question, cohort and cohort hash, baseline, single changed variable, branch, commit SHA, metrics, expected artifacts, and intended stage. The pre-run audit must pass before GPU approval. Artifacts are frozen before scoring, and the post-run audit verifies claims against the frozen evidence before the Orchestrator decides `GO`, `STOP`, or `PIVOT`.
