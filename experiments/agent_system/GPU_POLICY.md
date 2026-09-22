# ARC2 historical optional multi-agent GPU policy

> **Optional / legacy mode.** This is not a mandatory agent-routing workflow.
> The effective authorization rules are in root `AGENTS.md`; production release
> is governed by `docs/RELEASE_CHECKLIST.md`.

When this legacy mode is explicitly selected:

- Only one Kaggle GPU run may be active at a time.
- Every run requires a committed, exact-match approval record: experiment ID, branch, and commit SHA must match the run.
- Maximum requested wall time is 120 minutes; approved runs target less than two hours.
- Runs expected to exceed 60 minutes require MODEL_ESCALATION review before approval.
- Prefer the smallest experiment capable of changing the next decision.
- Use SMOKE (6-12 tasks, about 10-40 minutes), CONFIRM (20-30 tasks, about 30-90 minutes), then VALIDATE (40-60 tasks, below 120 minutes); do not expand automatically.
- Experiment Runners may not start GPU work without approval.
- After launch, rename or copy the approval state to `CONSUMED.json` in the relevant queue record.
- This repository provides governance only; it does not automatically launch GPU work.
