"""Fail closed unless orchestration is running from the permanent ARC2 worktree."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


EXPECTED_GIT_TOPLEVEL = "C:/Users/asus/Desktop/py/kaggle/ARC2"


class WorkspaceGuardError(RuntimeError):
    """The active Git worktree is not the authorized ARC2 orchestration root."""


GitRunner = Callable[..., subprocess.CompletedProcess[str]]


def verify_workspace_root(
    *,
    cwd: Path | None = None,
    runner: GitRunner = subprocess.run,
) -> dict[str, Any]:
    """Return a PASS receipt or raise when Git does not report the exact root."""

    working_directory = (cwd or Path.cwd()).resolve()
    try:
        result = runner(
            ["git", "-C", str(working_directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkspaceGuardError(f"WORKSPACE_GUARD_GIT_ERROR: {exc}") from exc

    stdout_lines = result.stdout.splitlines()
    observed = stdout_lines[0].strip() if len(stdout_lines) == 1 else ""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"git exited {result.returncode}"
        raise WorkspaceGuardError(f"WORKSPACE_GUARD_GIT_FAILURE: {detail}")
    if len(stdout_lines) != 1 or not observed:
        raise WorkspaceGuardError("WORKSPACE_GUARD_INVALID_GIT_TOPLEVEL_OUTPUT")
    if observed != EXPECTED_GIT_TOPLEVEL:
        raise WorkspaceGuardError(
            "WORKSPACE_GUARD_ROOT_MISMATCH: "
            f"expected={EXPECTED_GIT_TOPLEVEL!r} observed={observed!r}"
        )

    return {
        "status": "PASS",
        "expected_git_toplevel": EXPECTED_GIT_TOPLEVEL,
        "observed_git_toplevel": observed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for controller/director/supervisor startup gates."""

    if argv is None:
        argv = sys.argv[1:]
    if argv:
        print(
            json.dumps({"status": "FAIL", "error": "WORKSPACE_GUARD_TAKES_NO_ARGUMENTS"}),
            file=sys.stderr,
        )
        return 2
    try:
        receipt = verify_workspace_root()
    except WorkspaceGuardError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
