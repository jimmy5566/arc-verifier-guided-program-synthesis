# ARC2 Experiment Runner (optional historical role)

> **Optional / legacy mode.** This role is not required for ordinary
> single-Codex experiments. Root `AGENTS.md` is the effective default.

This role applies to both Experiment Runner A and Experiment Runner B. Each runner works only in its assigned worktree and implements only the assigned condition.

## Authorized autonomously

- Edit, create, and delete experiment files in the assigned worktree.
- Run CPU tests, `pytest`, and `py_compile`.
- Create artifacts.
- Build Kaggle notebooks.
- Stage and commit only the assigned branch; pushing still requires the
  explicit user authorization required by root `AGENTS.md`.

Run the CPU precheck before any approved GPU execution. Freeze candidate artifacts before scoring.

Within the assigned worktree, routine development is pre-authorized. Do not ask
the user for permission to edit project files, create scripts, run CPU tests,
run static checks, create artifacts, or commit the assigned branch. A push
still requires the root-rule authorization.

## Must stop

- GPU launch without an exact approval file.
- Competition submission.
- Changing another worktree.
- Modifying the production branch.
- Merge to `main`.
- Destructive Git actions.
- Experiment scope changes.
- Changing frozen scientific scope.
- Any action exceeding the approved GPU budget.

Do not tune extra variables. Implement only the assigned condition. Work autonomously until a final result, meaningful blocker, GPU approval gate, or competition-submission gate.
