# TurboDFS V5 reusable runtime assets

This project intentionally separates **Git source**, **Global immutable assets**,
and a **Pod-local rebuilt environment**.

## What Global Storage contains

`/workspace/arc2` is the established ARC2 Global root on the validated 3090
Pods.  After a passing V5 calibration, the promotion command creates:

- `models/qwen3_4b_grids15_sft139/`: the materialized offline Qwen checkpoint
  and tokenizer, with a file manifest;
- the 180 immutable adapters: hardlinked/reflinked/copied into
  `adapters/eval60_authoritative_greedy_v1/` only when their authoritative
  originals are outside Global.  When the authoritative run is already under
  the established Global root, the compact adapter manifest records those
  original immutable paths directly rather than duplicating roughly 177GiB;
- `assets/reference_bundle.tar.zst`: one compressed native-tokenizer reference
  bundle, not hundreds of small files;
- `turbodfs_v5/GLOBAL_ASSET_MANIFEST.json`,
  `GLOBAL_RUNTIME_PATHS.env`, and one 180-row adapter manifest.

Global Storage never contains a venv, `site-packages`, HF/PIP/Triton/Torch
caches, compiled kernels, a Git checkout, or per-cell TurboDFS traces.

## Clean Pod reconstruction

1. Mount the established Global root at `/workspace/arc2`.
2. Select the exact source commit recorded in `GLOBAL_ASSET_MANIFEST.json`.
3. Run from a fresh Pod-local location:

   ```bash
   export ARC2_SOURCE_REF=<exact-source-commit>
   export ARC2_REQUIRED_GPUS=1  # use 2 for the Eval60 collector
   bash scripts/bootstrap_turbodfs_v5_env.sh
   ```

The bootstrap verifies wheel/model hashes, recreates a Python 3.11 venv from
the frozen wheelhouse, stages the model to local storage, unpacks the one
reference bundle, validates BF16 xFormers attention, offline tokenizer/model
loading, one adapter at each depth (12/24/48), and the V5 decoder config.

## Adapter identity policy

The first Global promotion is the only full byte-level SHA256 audit of all 180
adapters.  It records the immutable paths, sizes and checkpoint hashes in the
compact Global manifest.  Later bootstrap and collector runs use that
attestation plus the small authoritative checkpoint manifest, fail closed on
missing/path/size/identity disagreement, and load the actual adapter tensors
normally.  They do **not** re-read the same roughly 177 GiB merely to recompute
SHA256 before every cell.  An exceptional storage-integrity investigation may
request `--reverify-adapters` explicitly.

## Resume contract

Use a Git checkout at the frozen commit and source
`/workspace/arc2/turbodfs_v5/GLOBAL_RUNTIME_PATHS.env`.  The runnable Eval60
state and block shards remain in the versioned run directory, whose SQLite
state database is authoritative.  Validate the immutable Global attestation
once per worker; never recreate TTT or Greedy merely to resume TurboDFS.
