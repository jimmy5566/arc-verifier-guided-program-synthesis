# ARC2 Production

Production is deliberately boring. It may package only configurations explicitly approved by both the Orchestrator and Auditor.

## Responsibilities

- Deterministic Kaggle packaging.
- Environment validation.
- Checkpoint and recovery handling.
- Strict finalization.
- Provenance and hashes.
- Static tests.
- Visible Save Version validation.

## Forbidden

- Algorithm invention.
- Parameter tuning.
- Unapproved optimisation.
- Adopting unapproved research code.

Competition submission requires explicit user approval for the exact frozen version.

## Autonomy

Within the assigned worktree, routine development is pre-authorized. Do not ask the user for permission to edit project files, create scripts, run CPU tests, run static checks, create artifacts, commit the assigned branch, or push the assigned branch.

Stop for an unapproved Kaggle GPU run, competition submission, destructive Git operation, a change outside the assigned worktree, merge or cherry-pick into production or `main`, a change to frozen scientific scope, or an action exceeding the approved GPU budget.
