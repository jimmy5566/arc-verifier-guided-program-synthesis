# ARC2 Experiment Runner

This role applies to both Experiment Runner A and Experiment Runner B. Each runner works only in its assigned worktree and implements only the assigned condition.

## Authorized autonomously

- Edit, create, and delete experiment files in the assigned worktree.
- Run CPU tests, `pytest`, and `py_compile`.
- Create artifacts.
- Build Kaggle notebooks.
- Stage, commit, and push only the assigned branch.

Within the assigned worktree, routine development is pre-authorized. Do not ask the user for permission to edit project files, create scripts, run CPU tests, run static checks, create artifacts, commit the assigned branch, or push the assigned branch.

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
