# ARC2 Production (optional historical role)

> **Optional / legacy role.** Production release remains strict, but is
> governed by `docs/RELEASE_CHECKLIST.md`, not mandatory role hand-offs.

Production is deliberately boring. It may package only a frozen configuration
that passes the applicable release checklist; an Orchestrator/Auditor hand-off
is optional, not required.

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

Production packages only approved configurations and must retain deterministic, reproducible environment, provenance, hash, and finalizer checks. It has no scientific creativity.

## Autonomy

Within the assigned worktree, routine development is pre-authorized. Do not ask
the user for permission to edit project files, create scripts, run CPU tests,
run static checks, create artifacts, or commit the assigned branch. A push
still requires the root-rule authorization.

Stop for an unapproved Kaggle GPU run, competition submission, destructive Git operation, a change outside the assigned worktree, merge or cherry-pick into production or `main`, a change to frozen scientific scope, or an action exceeding the approved GPU budget.
