# Forward Capability Repair ? Multistage Plan V1

**Objective:** preserve strong capabilities, repair weak atomic primitives, improve composition, then test separated ARC transfer. This is not a plan to restore same color alone.

| Stage | Starting checkpoint | Intervention | Expected evidence | Protection and evaluation | Budget / boundary |
| --- | --- | --- | --- | --- | --- |
| A ? anti-forgetting | `CAPABILITY_REPAIR_BASELINE_V1_V7` for both arms | Matched control/treatment; the only changed factor is the within-replay allocation of verified train-only `same color` examples. Treatment assigns 20% of the replay role to same color; control retains the prior non-same-color mixture. | Same-color preservation plus net paired improvement on connected components, inside/contains, difference, width, and orientation. | One prospectively frozen decoding/scoring condition for both arms; evaluate atomic, composition, and protected retention. `TARGET_DEV`, `RETENTION_SENTINEL`, and `FINAL_AUDIT` are never training inputs. | Joint cap 7,200 GPU seconds, max 3,600 per arm. Requires one distinct-protocol Director review before launch. |
| B ? atomic prerequisites | Valid Stage-A checkpoint selected after review | Increase only measured weak primitives: connected components, inside/contains, difference, width, orientation. | Progressive movement toward `>=0.75` with paired fixes exceeding harms. | Explicit replay coverage for every strong baseline capability, including same color; fixed DEV and retention measurements. | Use remaining global budget only after Stage-A actual charge and Stage-B review. |
| C ? composition | Valid Stage-B checkpoint selected after review | Increase relation-selector-action, conditional action, mask-set-construction, and related chains when prerequisites support it. | Composition gains that persist after accounting for atomic changes. | Continue protected-capability replay and retention evaluation; no FINAL_AUDIT. | Separate Stage-C review; global cumulative cap remains 28,800 seconds. |
| D ? ARC transfer | Selected model after prior selection is closed | No new training: evaluate on a separately frozen ARC development set. | Transfer beyond synthetic repair surfaces. | Development set is disjoint from training, DEV, retention, and FINAL_AUDIT; FINAL_AUDIT stays sealed until final selection. | Inference-only unless separately reviewed. |

## Evidence and constraints

- The stopped `FORWARD_TARGETED_CAPABILITY_REPAIR_V1` protocol, V7, R1, R2, and both batch diagnostics remain immutable.
- R1/R2 selected no `same color` family weight. The frozen source contains 250 train-only same-color examples, but R1/R2 per-family realized supervised-token counts were not recorded; R2 only recorded 214,455 total supervised tokens.
- The Stage-A hypothesis is causal only after the prospective matched comparison. The present evidence supports neither causality nor a new R3.
- The failed historical Batch1/Batch32 threshold remains unchanged. Stage A instead preregisters one common decoding condition for both arms, with the recorded fallback ladder and representative Batch1 check required by `AGENTS.md`.
