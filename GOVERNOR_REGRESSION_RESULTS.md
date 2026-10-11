# Governor regression results

Executed in the isolated worktree with the repository Python virtual
environment.  No Herdr, RunPod, model, GPU, or production state was touched.

```text
python -m pytest -q \
  tests/test_arc2_governor_controller_dispatch.py \
  tests/test_arc2_persistent_governor.py \
  tests/test_arc2_governor_infrastructure_triage.py \
  tests/test_arc2_governor_runpod_pty.py \
  tests/test_arc2_governor_idempotency.py

30 passed in 6.23s
```

`python -m py_compile scripts/arc2_governor.py
scripts/launch_e04_e_equal_slot_marker_replay_pilot.py` also passed, and
`git diff --check` reported no whitespace errors.

The new regression coverage verifies:

1. an identical action remains deduplicated after an old acknowledgement time;
2. a busy Controller retains its structured dispatch identity;
3. a fresh Governor reload does not repeat an acknowledged action;
4. an ambiguous Herdr timeout produces `UNKNOWN_DELIVERY` without a second prompt;
5. a new Controller stage receives one new delivery;
6. a stale workflow writer cannot overwrite a newer scientific transition;
7. an active remote job does not invoke Controller/launch work;
8. a terminal receipt produces one continuation and one receipt-consumption key;
9. `WAIT_REMOTE` polling invokes neither Controller nor Director; and
10. a live remote poll preserves existing scientific authorization fields.

Existing Controller, persistent-Governor, infrastructure-triage and PTY tests
also remain green.  The historical expectation that busy state overwrites a
structured dispatch with a string was updated to assert preservation instead.
## P0 follow-up regression

The Director delivery and cross-process state-revision corrections were tested
in the same isolated worktree.  No Herdr, RunPod, GPU, model, or production
Governor was invoked.

```text
python -m pytest -q \
  tests/test_arc2_governor_director_delivery.py \
  tests/test_arc2_governor_director_routing.py \
  tests/test_arc2_governor_idempotency.py \
  tests/test_arc2_governor_controller_dispatch.py \
  tests/test_arc2_persistent_governor.py \
  tests/test_arc2_governor_infrastructure_triage.py \
  tests/test_arc2_governor_runpod_pty.py

49 passed
```

The added coverage checks successful and confirmed-non-delivery Director
paths, ambiguous timeout without re-prompt, process-restart persistence, late
SHA-bound response consumption, and one-time response consumption.  It also
runs two independent Python writer processes: a stale writer fails with a
revision conflict after a separately loaded Controller transition is committed.
`py_compile` and `git diff --check` passed.
