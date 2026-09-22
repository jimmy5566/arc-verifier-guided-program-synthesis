# Historical optional multi-agent material

The files in this directory preserve prior ARC2 orchestration, runner,
auditor, and GPU-approval procedures as historical evidence and optional
isolation tooling. They are **not** required for normal development.

The effective project rule set is the repository-root `AGENTS.md`: default
development is one Codex agent in the current repository/worktree. Use
worktrees or parallel agents only when they provide clear isolation or
throughput value. Production release remains governed by
`docs/RELEASE_CHECKLIST.md`.
