# Public TurboDFS audit for Eval60 joint screen V2 public

## Verified source

- **Repository:** `https://github.com/1ytic/NVARC`
- **Commit:** `846d0198efa752534594e321fc3289fc0a06c657`
- **Notebook:** `ARC-AGI1/002_ivan_arc1.ipynb`
- **Notebook SHA256:** `24e57f6ae39e63d0041641ae2302a40f7afb9c2973420fb8db05054028537e52`
- **Actual inspected source:** notebook cells which write `arc_solver.py` and `arc_loader.py`; specifically `turbo_dfs`, `inference_turbo_dfs`, and `QwenFormatter.max_new_tokens`.
- **Local implementation:** `src/inference/nvarc_turbodfs_opt.py`.

The public code was inspected from a local clone pinned at the stated commit, not inferred from comments. Its decoder uses a compact ARC token vocabulary, recursive batched live-lane expansion with `past_key_values`, float32 CPU log-softmax cumulative-NLL pruning, score-ordered frontier traversal, a 540-second per-call search stop, and no forward-pass cap.

## Frozen decoder decision

`TURBODFS_OPT_V1` for this new run retains the verified public batching/KV/NLL/EOS/max-length semantics. It removes the non-public 256-forward cap and restores the public 540-second stopping limit. The local cap of four retained completed candidates and deterministic score/token tie ordering are explicitly frozen search semantics, not claimed as public speed optimizations. They are retained to bound storage and candidate handling; frozen V1 traces showed no evidence that the candidate cap caused zero-complete failures.

The full classification is in `public_turbodfs_feature_matrix.csv`. No Gold labels, answers, or post-hoc branch choices were used in this audit or decoder configuration.
