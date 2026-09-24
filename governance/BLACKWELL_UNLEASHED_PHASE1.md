# Blackwell-unleashed Phase 1

This is a bounded execution-only Eval3 benchmark.  The immutable queue is
`governance/queues/5090-unleashed-phase1-v1.json`:

- GPU 0: serial generation, then batch 4.
- GPU 1: batch 2, then a separate batch-4 repeat.
- TTT remains serial and unchanged for every task.
- Candidate pools freeze before the CPU-only Any-of-K scorer opens solutions.

The launcher is deliberately fail-closed.  It first calls
`bootstrap_5090_research_env.sh`, which requires exactly two RTX 5090 sm_120
devices plus verified portable assets on `/workspace-global/arc2`.  It then
uses `CUDA_VISIBLE_DEVICES=<physical GPU>` only in each child, while CUDA inside
that child remains `cuda:0`.  This prevents GPU1 work from silently landing on
physical GPU0.

Run only after the portable assets have been seeded and verified:

```bash
/root/arc-runtime/env/5090-blackwell-env-v2/bin/python \
  /root/arc-runtime/arc2/scripts/run_5090_blackwell_unleashed_phase1_queue.py \
  --queue-config /root/arc-runtime/arc2/governance/queues/5090-unleashed-phase1-v1.json \
  --bootstrap-script /workspace-global/arc2/bootstrap/bootstrap_5090_research_env.sh \
  --sync-script /workspace-global/arc2/bootstrap/sync_arc2_run_to_persistent.sh \
  --manifest /workspace-global/arc2/benchmarks/legacy-eval3/eval3_manifest.json \
  --reference-config /workspace-global/arc2/benchmarks/legacy-eval3/reference_ttt_config_frozen.json \
  --challenge-path /path/to/arc-agi_evaluation_challenges.json \
  --solutions-path /path/to/arc-agi_evaluation_solutions.json \
  --model-path /root/arc-runtime/model-stage/qwen3_4b_grids15_sft139 \
  --model-manifest /workspace-global/arc2/models/qwen3_4b_grids15_sft139/model_manifest.json \
  --native-config-dir /root/arc-runtime/arc2/configs/nvarc_native_846d0198 \
  --environment-lock /workspace-global/arc2/env/environment_lock.txt \
  --environment-manifest /workspace-global/arc2/env/environment_manifest.json \
  --ptxas-path /usr/local/cuda-12.8/bin/ptxas \
  --repo-url <canonical-ARC2-repository-url>
```

Each successful run is atomically synced to
`/workspace-global/arc2/frozen-runs/<run_id>/` with the mandated manifest,
config, environment, candidates, telemetry, events, hashes and post-freeze
evaluation report.  A separate immutable queue-summary run stores Jaccards and
the promotion decision once all four runs have completed.

Candidate drift, Jaccard differences and Any-of-K changes remain diagnostics,
not stop conditions.  OOM, non-finite/integrity failure, malformed artifacts,
environment failure and persistent-sync failure stop subsequent dispatch.

## Single-GPU ephemeral amendment

`5090-unleashed-phase1-single-gpu-v1.json` amends the queue for a verified
single RTX 5090 Pod without a Global Volume.  It runs the same four conditions
on physical GPU0 in this strict order: batch 1, batch 2, batch 4, batch 4
repeat.  It does not use `/workspace-global`.

The local controller
`scripts/run_5090_blackwell_unleashed_phase1_offpod_controller.py` is the only
supported dispatcher for this amendment.  It runs one remote condition, waits
for the remote atomic candidate freeze and SHA256 manifest, transfers the run
as a terminal-safe base64/tar archive, verifies the archive and candidate hash
on the local PC, scores locally only after that freeze, then dispatches the
next condition without human interpretation.  A failed transfer or hash check
stops the queue before more GPU time is used.  Eval60 remains out of scope.
