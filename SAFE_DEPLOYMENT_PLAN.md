# Safe deployment plan

## Preconditions

1. RUN_003 has reached terminal state and its terminal receipt has been
   preserved.  Do not apply this branch to the live Governor while it is
   monitoring RUN_003.
2. Confirm exactly one Governor/watchdog owns the durable workflow state and no
   Controller turn is currently writing that state.
3. Confirm the deployment checkout is clean and record the current workflow
   state SHA-256 as a rollback artifact.
4. Review `GOVERNOR_IDEMPOTENCY_PATCH.diff` and the CPU regression result.

## Deployment sequence

1. Stop only the live Governor after terminal receipt consumption; do not stop
   or replace any remote scientific job.
2. Fast-forward the approved branch to the reviewed fix commit.  Do not modify
   a SHA-bound experiment worktree or historical binding.
3. Run the CPU regression suite from `GOVERNOR_REGRESSION_RESULTS.md` against
   the deployment checkout.
4. Restart one Governor/watchdog pair using the existing hidden-window service
   mechanism.  Verify its first cycle only reads the existing terminal or next
   Controller state.
5. Verify `state_revision` increases on a benign state update and that a
   duplicate semantic action reports `CONTROLLER_ALREADY_DISPATCHED` rather
   than prompting Herdr.

## Rollback

If CPU tests fail or the service reports `StateConflict` repeatedly, stop the
Governor, restore the recorded checkout and durable state backup, then restart
one Governor only.  A conflict is a safe failure: it has not overwritten the
newer scientific transition.

## Deployment boundary

This worktree has not been pushed or deployed.  The future PTY change applies
only to launches bound after review; RUN_003 continues to use its immutable
remote snapshot and is outside this patch's execution path.
