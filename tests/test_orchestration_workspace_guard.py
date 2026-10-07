from pathlib import Path
from subprocess import CompletedProcess

import pytest

from orchestration.workspace_guard import (
    EXPECTED_GIT_TOPLEVEL,
    WorkspaceGuardError,
    verify_workspace_root,
)


def _runner(*, stdout: str, stderr: str = "", returncode: int = 0):
    def run(*args, **kwargs):
        return CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def test_exact_authorized_root_passes() -> None:
    receipt = verify_workspace_root(
        cwd=Path("."),
        runner=_runner(stdout=f"{EXPECTED_GIT_TOPLEVEL}\n"),
    )
    assert receipt == {
        "status": "PASS",
        "expected_git_toplevel": EXPECTED_GIT_TOPLEVEL,
        "observed_git_toplevel": EXPECTED_GIT_TOPLEVEL,
    }


@pytest.mark.parametrize(
    "observed",
    [
        "C:/Users/asus/Desktop/py/kaggle/ARC2-copy\n",
        "c:/Users/asus/Desktop/py/kaggle/ARC2\n",
        "C:\\Users\\asus\\Desktop\\py\\kaggle\\ARC2\n",
    ],
)
def test_non_exact_root_fails_closed(observed: str) -> None:
    with pytest.raises(WorkspaceGuardError, match="WORKSPACE_GUARD_ROOT_MISMATCH"):
        verify_workspace_root(cwd=Path("."), runner=_runner(stdout=observed))


def test_git_failure_fails_closed() -> None:
    with pytest.raises(WorkspaceGuardError, match="WORKSPACE_GUARD_GIT_FAILURE"):
        verify_workspace_root(
            cwd=Path("."),
            runner=_runner(stdout="", stderr="not a repository", returncode=128),
        )


@pytest.mark.parametrize("stdout", ["", "one\ntwo\n"])
def test_invalid_git_output_fails_closed(stdout: str) -> None:
    with pytest.raises(WorkspaceGuardError, match="WORKSPACE_GUARD_INVALID_GIT_TOPLEVEL_OUTPUT"):
        verify_workspace_root(cwd=Path("."), runner=_runner(stdout=stdout))
