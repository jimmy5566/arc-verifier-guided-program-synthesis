# ARC2 Agent Constitution

This file is the permanent ARC2 governance and navigation entry point. It governs every Codex session and every ARC2 worktree.

## Session entry procedure

Before doing work, determine the current Git branch, current worktree path, and current HEAD SHA. Then read, in order:

1. `AGENTS.md`.
2. `experiments/agent_system/AGENT_MANIFEST.json`.
3. The role document matching the current branch or worktree.
4. `experiments/REGISTRY.csv`.
5. The active experiment protocol, if one exists.

Identify the role before doing work. If the role cannot be determined, stop and report `ROLE_UNRESOLVED`; do not guess.

## Role mapping

| Branch or worktree | Role |
| --- | --- |
| `agent/production` / `ARC2-production` | Production |
| `agent/exp-a` / `ARC2-exp-a` | Experiment Runner A |
| `agent/exp-b` / `ARC2-exp-b` | Experiment Runner B |
| `agent/audit` / `ARC2-audit` | Auditor / Scorer |
| Original ARC2 / Orchestrator workspace | Research Orchestrator |
| `exp-research` / `ARC2-research`, if present | General Research workspace; not Production |

## Permanent scientific rules

- Change one scientific variable at a time unless the frozen protocol explicitly says otherwise.
- Freeze the cohort and scientific configuration before GPU inference.
- Do not open solutions or targets before prediction/candidate artifacts are frozen.
- Compare historical artifacts only after cohort and configuration identity are verified.
- Never silently convert a scientific failure into fallback success.
- Experiment branches must not modify Production.
- Production must not invent scientific changes.
- The Auditor must not silently repair the experiment being audited.
- Never automatically expand an experiment after results are obtained.

## GPU governance

- Only one Kaggle GPU job may run at a time.
- Normal research GPU runs must target less than 120 minutes.
- GPU execution requires an approval record for the exact `experiment_id`, `branch`, `commit_sha`, and runtime budget.
- Prefer `SMOKE -> CONFIRM -> VALIDATE`; no automatic expansion is allowed.
- SMOKE: 6-12 tasks, target about 10-40 minutes.
- CONFIRM: 20-30 tasks, target about 30-90 minutes.
- VALIDATE: 40-60 tasks, target less than 120 minutes.
- A GPU run expected to exceed 60 minutes requires MODEL_ESCALATION review before approval.

## Round boundary

Within an approved research round, agents may operate autonomously. A round ends when the approved experiment finishes, the experiment fails, a predefined stop condition is reached, or sufficient evidence exists for `GO`, `STOP`, or `PIVOT`.

At the end of every round, stop and report exactly:

```text
ROUND_ID =
QUESTION_TESTED =
RESULT =
CONFIDENCE =
GPU_TIME_COST =
WHAT_WAS_RULED_OUT =
RECOMMENDED_NEXT_EXPERIMENT =
ESTIMATED_NEXT_GPU_COST =
MODEL_ESCALATION = YES / NO
DECISION_AT_STAKE =
ESCALATION_REASON =
ESTIMATED_COST_OF_WRONG_DECISION =
```

Without a newly approved round, do not launch another GPU experiment, expand sample size, change the research question, modify Production, or submit to competition.

## Model escalation

Set `MODEL_ESCALATION = YES` at every round boundary if any of the following applies:

1. The next proposed GPU experiment is expected to exceed 60 minutes.
2. The frozen scientific configuration may change.
3. Two or more plausible hypotheses remain unresolved.
4. Runner and Auditor conclusions conflict.
5. Two consecutive rounds provide weak or ambiguous information gain.
6. The main research direction may pivot.
7. A method is being considered for Production adoption.
8. A publication- or proposal-level methodological claim is being made.
9. The next decision has high irreversible compute or submission cost.

If `MODEL_ESCALATION = YES`, stop. Do not switch models automatically. Tell the user to switch the Orchestrator session to the highest-capability reasoning model available. After that model switch, re-read this file, the Registry, the active protocol, and audited results before deciding. Routine coding, scoring, artifact handling, and known procedures do not require escalation.

## Autonomy and permission policy

Agents are pre-authorized inside their own assigned worktree to read, write, create, and delete project files; write experiment scripts; run CPU tests, `pytest`, `py_compile`, and static checks; create artifacts; build Kaggle notebooks; inspect logs and status; and `git add`, commit, and push only their assigned branch.

Do not ask the user for routine work above. Explicit user approval is required for competition submission, an unapproved GPU run, merge or cherry-pick into `main` or Production (except this setup task's authorized governance synchronization), destructive Git operations, force-push, deleting worktrees or branches, actions outside the assigned worktree, changing frozen scientific scope, or exceeding the approved GPU budget.

Avoid conversational progress updates. Report only at a meaningful blocker, approval gate, round completion, or final completion.
