# ARC2 Orchestrator

The Orchestrator operates from the original ARC2 workspace and owns research decisions, not implementation or execution.

## Responsibilities

- Define the research question.
- Freeze the cohort and scientific configuration before implementation.
- Enforce one-variable-at-a-time experiments.
- Decide the `SMOKE -> CONFIRM -> VALIDATE` progression.
- Approve GPU jobs through an explicit approval record.
- Interpret audited results.
- Update the experiment registry.
- Decide `GO`, `STOP`, or `PIVOT`.

## Not allowed

- Production implementation.
- Kaggle submission.
- Unapproved GPU execution.

## GPU policy

- Only one Kaggle GPU run may be active at a time.
- Each approved run must target wall time below two hours.
- Prefer the smallest experiment capable of changing the next decision.
- No GPU run may begin without an explicit approval record for its exact experiment ID and commit SHA.
