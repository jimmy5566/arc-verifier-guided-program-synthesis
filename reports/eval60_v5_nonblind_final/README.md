# Eval60 V5 recovered nonblind final handoff

## Status

- Generation completed: `1068/1068` cells, with no active GPU worker.
- Execution mode: `NONBLIND_USER_AUTHORIZED_CONTINUATION`.
- Generator use of Gold: `false`; Gold was used only for authorized interim analysis after generation began.
- Database: Pod-local recovered SQLite; `PRAGMA quick_check = ok` at the final snapshot.
- Standard V5 freeze: **blocked**, not claimed complete.

## Frozen evidence

- Final lightweight checkpoint: `reports/eval60_v5_nonblind_interim/checkpoints/HEAVY_189_DONE/`.
- Checkpoint commit: `10ec3f79ce80c4a843806416c4530079f55a66af`.
- Final snapshot: `1068/1068 DONE`; `879` fast-pass and `189` heavy-repair cells.
- Record-validated analysis rows: `880`; `188` recovered DB DONE rows were excluded because their immutable candidate-record association was missing or ambiguous.
- Reported pool quantities are conservative lower bounds: V5 `28/89`; Greedy union V5 `33/89`.
- Top-2: `NOT_COMPUTABLE` because no frozen V5 selector evidence exists.

## Standard-freeze blocker

The existing compaction finalizer stopped at `1818057f:o0:d12` with:

```
RuntimeError: DONE cells without a valid temp group for 1818057f o0 d12
```

The recovered database contains lane-to-temp-path collisions for that output and
`284` DONE rows without a direct `temp_path` (they may instead refer to a
compacted shard). Therefore it would be incorrect to assert the normal
`V5_GENERATION_FROZEN.flag` or an all-candidate artifact manifest. No candidate
or state row was modified to make this pass.

## Preservation inventory

| Asset | Location | Size | Preservation status |
| --- | --- | ---: | --- |
| Raw V5 candidate traces | `/workspace/arc2/active_runs/eval60_v5_vs_greedy_4worker_v1/tmp` | 3.8 GiB | Pod-resident; not pushed to ordinary Git |
| Runtime generation inputs | `/workspace/arc2/active_runs/eval60_v5_vs_greedy_4worker_v1/generation_inputs` | 1002 KiB | Pod-resident, identities below |
| Recovered state database | `/root/arc-runtime-turbodfs-v5-benchmark/state/eval60_v5_nonblind_recovered.sqlite` | 1,056,768 bytes | Static byte-verified snapshot copied to `/workspace/arc2/active_runs/eval60_v5_vs_greedy_4worker_v1/recovered_state_snapshot/`; ordinary Git upload not performed |
| Compact reports and provenance | this Git directory plus the HEAVY checkpoints | small | pushed to GitHub |

Normal Git/LFS tracking was not configured for the 3.8 GiB candidate directory;
it was intentionally not uploaded as an unreviewed large blob. A separate
durable-artifact transfer decision is required before deleting the Pod.

## Identities

- V5 config SHA256: `e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0`
- TTT config SHA256: `8968f67e5a5c8c1f838dc1a45120d052527d65e95188cd9e155426ba56481486`
- Adapter/checkpoint manifest SHA256: `e3e95956b0c17e3017b3bb99999c53bfc5307908df68c43fc6641e8b1459c2a6`
- Current recovered DB SHA256: `12529cc829509f224f466446ccf08663083dd7fd5aea849f73cb2e449d7ac485`
- Static `/workspace` snapshot SHA256: `12529cc829509f224f466446ccf08663083dd7fd5aea849f73cb2e449d7ac485` (byte-identical). The FUSE mount did not support timestamp preservation during copy; content verification passed.

