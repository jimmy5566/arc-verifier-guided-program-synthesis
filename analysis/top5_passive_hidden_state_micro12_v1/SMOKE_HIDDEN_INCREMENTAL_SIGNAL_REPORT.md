# Smoke hidden incremental-signal audit

This is a **no-Gold** directional diagnostic over the three frozen Gate1 smoke outputs.

- Records: **16,130**
- Hidden/frontier lineage join: **100% on all three outputs**
- Decision: **SMOKE_HIDDEN_INCREMENTAL_SIGNAL_NOT_ESTABLISHED**

## Main result

Adding the three passive parent-hidden layers to frozen scalar telemetry did not produce stable out-of-task improvement:

- Eventually-expanded proxy: mean Delta AUC = **-0.00006**
- Expanded-within-64 proxy: mean Delta AUC = **0.00000**
- >=8 expanded descendants proxy: mean Delta AUC = **+0.01855**, but the combined model improved only 2/3 held-out tasks.

A notable diagnostic is layer 23 alone on the descendant-fertility proxy:

- 1818057f:o0: **+0.04153 AUC**
- 36a08778:o1: **+0.01845 AUC**
- 80a900e0:o0: **+0.01585 AUC**

Thus the middle-layer representation appears to contain some transferable branch-context/fertility information, but the full passive hidden representation is not yet a stable selector.

## Representation anatomy

Approximate effective rank of the sampled hidden representations:

| Output | Layer 11 | Layer 23 | Layer 35 |
| --- | ---: | ---: | ---: |
| 1818057f:o0 | 15.23 | 36.77 | 4.05 |
| 36a08778:o1 | 15.76 | 49.12 | 6.45 |
| 80a900e0:o0 | 13.64 | 37.41 | 6.30 |

Layer 23 carries substantially richer variation than layers 11 or 35.

Within a single capture step, hidden vectors are effectively identical for sibling candidates, as expected from the passive **parent-prefix** representation. Therefore this representation cannot by itself distinguish sibling child tokens; it only supplies branch/context information.

## Scientific interpretation

This diagnostic does **not** use Gold and therefore cannot establish whether hidden states identify the correct ARC path.

It does show that:
1. passive parent hidden states are non-degenerate;
2. layer 23 contains some transferable information about future subtree activity;
3. the current three-layer scalar+hidden selector is not robust enough to justify a live search-policy change.

For a definitive answer about Gold discrimination, use the preregistered post-freeze Micro12 Gold probe.
