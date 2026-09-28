# Eval60 full-release archive restore

Large ARC2 evidence is stored as GitHub Release assets, never normal Git blobs.
The archive builder streams `tar | zstd -3 | split` with deterministic 1.5 GiB
parts. Its file-level hashes live in `artifacts/eval60_full_archive/`.

Examples:

```bash
scripts/restore_eval60_full_archive.sh --destination /workspace/arc2/restored --adapters
scripts/restore_eval60_full_archive.sh --destination /workspace/arc2/restored --greedy --v5
```

The base-model release is intentionally unavailable unless the recorded model
redistribution gate is `ALLOWED`. In that case, restore the exact upstream
model using the identifier and hashes recorded in the archive manifest instead.

V5 release data is a `snapshot-1068` archive, not a standard final freeze: its
recovered scheduler state has known candidate-linkage ambiguity. The corrupted
FUSE `run_state.sqlite` is explicitly excluded from normal restoration.
