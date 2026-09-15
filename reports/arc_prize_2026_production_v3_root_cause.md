# ARC Prize 2026 Version 3 production incident

## Evidence

The downloaded Version 3 log contains one production-path event only:

```json
{"event":"FAST_COMMIT_MODE","gpu_workers_started":0,"model_loaded":false,"task_count":240,"validation":"PASS"}
```

It contains neither `COMPETITION_RERUN_MODE` nor a Dynamic-B generation,
selection, or worker-start event.  The downloaded `submission.json` also has
259 test outputs with `attempt_1 == attempt_2` and its SHA-256 equals the
FAST_COMMIT log value.  It is therefore the identity-input fallback generated
by the commit branch, rather than a model prediction artifact.

## Root cause

**Confirmed: A — the submitted execution did not enter full production mode.**
The Version 3 notebook selected its branch solely from
`KAGGLE_IS_COMPETITION_RERUN`.  In the execution that produced the scored
artifact this value was absent or false, and the ordinary fast-commit branch
wrote the fallback to `/kaggle/working/submission.json`.

This is a production-pipeline failure, not evidence about Qwen, A, or B
accuracy.

## Contributing safety defects

* **B is also possible in a partial run.**  The finalizer accepts a missing or
  partial B artifact, but then drops directly to identity fallback.  It did
  not preserve completed A candidates for those tasks.
* **C is possible.**  The former deadline was advisory: the notebook used
  blocking `subprocess.run`, so it could not enforce the hard-finalization
  reserve while generation or reranking was stuck.
* **D is not supported by current evidence.**  The artifact assembled is the
  fast-commit artifact the notebook explicitly wrote; it was not a stale or
  unrelated file.

## Required corrective controls

Version 4 must log a sanitized execution-environment audit, distinguish
`FAST_COMMIT_MODE — NOT A COMPETITION PREDICTION` from
`PRODUCTION_INFERENCE_ACTIVE`, use B → A → identity fallback provenance, and
enforce a parent-owned hard process watchdog with time reserved for assembly.
