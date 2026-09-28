# Eval60 authoritative Greedy completion

Status: **PARTIAL — authoritative Greedy surface complete; TurboDFS V3 did not
pass its pre-registered micro gate.**

The exact Eval60 cohort contains 60 tasks and 89 outputs.  The frozen Greedy
surface contains all 1,068 cells (three retained TTT depths by four views per
output) and all 180 adapter checkpoints.  A post-run checksum audit verified
all checkpoint bytes and all required Greedy token-telemetry fields.

The public-parity TurboDFS V3 micro calibration was target-blind.  It produced
three complete valid candidates in six cells, a 50% zero-complete rate, and
therefore failed the frozen gate.  Full V3 calibration and TurboDFS blocks were
not started.  This is a mechanism-gate result, not a claim about output
accuracy.

The official evaluation solutions were read only after the Greedy generation
freeze.  They yield a retrospective Greedy pool oracle of 29/89.  Raw cells,
adapters, telemetry, hashes, Gold labels, and the morning handoff remain at the
run directory recorded in `RUN_COMPLETION_MANIFEST.json`; they are deliberately
not tracked in Git.
