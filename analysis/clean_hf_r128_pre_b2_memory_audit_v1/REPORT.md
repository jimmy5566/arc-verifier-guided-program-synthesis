# CLEAN_HF_R128_PRE_B2_MEMORY_AUDIT_V1

Target-blind, fresh-process, four-view R128 audit through exactly one physical B2 forward.

| checkpoint | allocated GiB | reserved GiB | live KV bytes |
|---|---:|---:|---:|
| P1 model + adapter | 7.8144 | 8.8262 | 0 |
| P5 four prefills | 8.8819 | 9.3105 | 1137180672 |
| P6 before first B2 | 8.8819 | 9.3105 | 1137180672 |
| B5 before B2 forward | 9.4114 | 9.7402 | 1137180672 |
| B6 after B2 forward | 9.9412 | 10.4434 | 1137180672 |
| B8 after split/reconstruct | 9.9412 | 10.4434 | 1137180672 |
| B9 transient references released | 9.4117 | 10.4434 | 1137180672 |
| B10 logical replies resumed | 9.4117 | 10.4434 | 1706065920 |

## Attribution

- P6 unattributed allocation: 75362048 bytes (0.0702 GiB).
- Merge before forward: 568590336 bytes (0.5295 GiB).
- `DynamicCache.from_legacy_cache`: ZERO_COPY_WRAPPER (72/72 storages shared).
- First B2 forward delta: 568885760 bytes (0.5298 GiB).
- Split independent-cache copies: 568885248 bytes (0.5298 GiB), sharing 0 merged storages.
- After B9 releases the merged temporary representation, allocation drops 568590336 bytes.  After B10 the live-KV set remains higher by 568885248 bytes: the split copies are now owned by resumed DFS states.

## Classification

`SPLIT_CONTIGUOUS_COPY` is the primary explanation.  The historical 19.65-GiB allocated OOM exceeds fresh P6 by about 11562167706 bytes, equivalent to 20.32 first-B2 split-copy increments.  The historical 3.59-GiB reserved-but-unallocated value is a secondary allocator-fragmentation observation.  No committed dedicated B2 micro/lane-swap/repeat runner was found, so stale micro-gate references cannot be quantitatively reconstructed.

The corrected R32 four-view+B2 projection uses the maximum simultaneous four-cell snapshot `F96`, not END: 10626989312 bytes (9.8972 GiB), `fits_24gb=True`.  That projection covers a single B2 pair only; it does not prove a full R128 dynamic B2 search fits 24GB.
