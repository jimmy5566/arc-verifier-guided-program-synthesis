# Protocol deviation: early partial post-freeze scoring

On 2026-10-02, before the 18-output generation freeze but after the S/M outputs had atomically completed, the user explicitly authorized one CPU-only, post-hoc Gold score of those completed S/M pools.  It only read the frozen output checkpoints and the evaluation solutions.  It did not load a model, use GPU, modify a candidate pool, modify an output checkpoint, change the remaining L generation, or change any scientific configuration.

The original global integrity condition, “Gold is first accessed only after the full 18-output generation freeze,” is therefore **not established** for this experiment.  The automatic final scorer did run after `GENERATION_FREEZE.json` and `GENERATION_HASH_VERIFICATION.json` passed; its local indicator does not override the earlier authorized access.

All final analysis from this pilot must be labelled `EARLY_PARTIAL_POSTHOC_GOLD_SCORE` and must not be represented as strictly target-blind evidence.
