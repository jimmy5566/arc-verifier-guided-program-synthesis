# Augmentation candidate screening V1

## Result

- **Public classification:** `PUBLIC_DFS_AUG16_CONFIRMED`.
- **Research classification:** `RESEARCH_AUG16_CANDIDATES_FROZEN`.
- The public source configures 16 inference/DFS subkeys per output: eight D4 geometries times two seeded full-colour permutations. They are decoded in four physical batches of four. This is distinct from the public TTT expansion (eight geometries times sixteen seeded colour samples = 128 records) and from rescoring (eight geometries times one colour sample = eight views).
- Public `16` is not a portable canonical set of 16 static recipes: source code does not pin NumPy, and every augmented view's train-pair ordering is drawn according to the task's number of demonstrations. Candidate rescoring also derives its seed from `hash(bk)`, whose process salt is not pinned in source.

## Research pool

The frozen 24-candidate pool deliberately keeps `PROJECT_RESEARCH_AUG4 ⊂ AUG8 ⊂ AUG16 ⊂ AUG24`. Its AUG16 adds eight target-blind, exactly invertible representation conditions to D4; it is **not** the public NVARC AUG16.

The redundancy screen used eight challenge-only representative tasks chosen by a SHA-256 rule. It verifies serialized prompt and transformed-test-input identities. Exact Qwen token lengths are marked unavailable because no tokenizer/model was loaded; byte lengths are recorded as the CPU-only representation measure. Per-task duplicate variants caused by a one-example train set are reported but retained when the same candidate differs on other tasks.

## Next step

No GPU work occurred. If a new GPU experiment is explicitly authorized, the appropriate next width is `DYNAMIC_B16`, using the frozen **project** AUG16 definition and an independently held-out screening protocol.
