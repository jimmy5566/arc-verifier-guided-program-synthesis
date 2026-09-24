# ARC2 governed experiment operations

## 1. Prepare data

Keep the raw challenge read-only. Create an inference bundle with train
inputs/outputs and test inputs only. If local targets are supplied, write them
to a physically separate path.

```powershell
python scripts/prepare_governance_data.py prepare `
  --challenges <raw-challenges.json> `
  --cohort governance/registry/cohorts/eval60.json `
  --source-dataset <source-record.json> `
  --bundle-output <inference-bundle>
```

Validate the resulting bundle with the `validate` subcommand. Platform mounts
that expose solutions provide policy isolation, not physical isolation; record
that distinction honestly.

## 2. Resolve and validate

Copy `governance/configs/fixed4plus4_d1_example.json`, change only the
predeclared experiment/execution fields, and keep every scientific field
explicit. Then run:

```powershell
python scripts/run_experiment_workbench.py validate --config <config.json>
python scripts/run_experiment_workbench.py dry-run --config <config.json>
python scripts/run_experiment_workbench.py prepare --config <config.json>
```

Validation, dry-run, and preparation are CPU-only. Dry-run validates the
inference-only challenge without importing the shared solver. A new run ID creates a new directory and cannot
overwrite a completed or partial run.

## 3. Authorize and run

GPU execution is a separate explicitly authorized action. The thin workbench
imports the existing `run_live` solver only for the `run` subcommand:

```powershell
python scripts/run_experiment_workbench.py run --run-dir <runs/run-id>
```

The result freezes `candidates_frozen.json`, `scores_frozen.json`, and
`predictions_frozen.json`. It does not create or submit `submission.json`.
If an attempt reaches `FAILED` or is explicitly recorded `INCOMPLETE`, rerun
the same command against the same run directory. The workbench rejects changed
resolved config bytes and delegates exact checkpoint compatibility to the
existing shared solver before reuse.

## 4. Score locally

Only after `COMPLETE_FROZEN`, use a separate local evaluation artifact:

```powershell
python scripts/run_experiment_workbench.py score `
  --run-dir <runs/run-id> --evaluation-targets <local-targets.json>
```

The run becomes `COMPLETE_SCORED`; exposure remains recorded independently.
`NO_GAIN` is a conclusion only after complete scoring, never a timeout label.

## 5. Plan reuse

Use stage-specific compatibility keys from `governance.workbench.plan_reuse`.
Only `REUSE_EXACT` is reusable. `UNKNOWN`, `COMPUTE_REQUIRED`, and
`INCOMPATIBLE` must not load the artifact as evidence. Missing adapter
checkpoints mean adaptation must be rebuilt.

## 6. Promote locally

After explicit approval, create a local release-candidate manifest:

```powershell
python scripts/promote_experiment_candidate.py `
  --run-dir <runs/run-id> --approval-record <approval.json> `
  --output <release-candidate.json>
```

Promotion references the exact code/config and frozen prediction hashes. It
does not publish a dataset, push a notebook, start a remote run, or submit.
