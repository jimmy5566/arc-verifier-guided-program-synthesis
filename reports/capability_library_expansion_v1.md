# Capability Library Expansion V1

这是与 Recognition 隔离的 development-only solution-aware representability audit。候选仅从 train pairs 归纳，并且必须同时 train exact 与 test exact 才计入。

- Before: **0/30**
- After: **1/30 (3.3%)**
- Status: **CAPABILITY_STILL_BLOCKED**

## Family coverage

| Family | Covered | Total |
| --- | ---: | ---: |
| iteration_repeat | 0 | 15 |
| relational | 0 | 8 |
| conditional | 0 | 4 |
| composition | 0 | 2 |
| output_construction | 1 | 1 |

## Added generic capabilities

| Capability | Test-exact coverage | Marginal gain |
| --- | ---: | ---: |
| CAP_DIAGONAL_SEQUENCE_TRAIL_V1 | 1 | 1 |
| CAP_REPEAT_COPY_TRANSLATION_V1 | 0 | 0 |
| CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1 | 0 | 0 |
| CAP_TILE_MASK_AT_ANCHORS_V1 | 0 | 0 |
| CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1 | 0 | 0 |

所有 per-task witness、grid 与 test target 仅保存在 ignored 私有 artifact。
