# ARC2 GPU Governance Policy

- Only one Kaggle GPU run may be active at a time.
- Every run requires a committed, exact-match approval record: experiment ID, branch, and commit SHA must match the run.
- Maximum requested wall time is 120 minutes; approved runs target less than two hours.
- Runs expected to exceed 60 minutes require MODEL_ESCALATION review before approval.
- Prefer the smallest experiment capable of changing the next decision.
- Use SMOKE (6-12 tasks, about 10-40 minutes), CONFIRM (20-30 tasks, about 30-90 minutes), then VALIDATE (40-60 tasks, below 120 minutes); do not expand automatically.
- Experiment Runners may not start GPU work without approval.
- After launch, rename or copy the approval state to `CONSUMED.json` in the relevant queue record.
- This repository provides governance only; it does not automatically launch GPU work.
