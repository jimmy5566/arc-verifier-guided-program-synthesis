# G0.5 incremental KV replay parity

STATUS: PASS_NUMERICAL
Tolerance: 0.05

The incremental path uses the same `model.generate` KV-cache loop as the historical Greedy run. A diagnostics-only logits processor captures each raw distribution and forces the frozen historical token; no free decoding is performed. No custom position IDs are supplied. Full TF is retained only as a diagnostic control.

| depth | task/output/view | historical LP | incremental LP | full-TF LP | inc abs delta | full abs delta | inc max native delta | full max native delta | inc strict parity | full strict parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 12 | 0934a4d8:o0 / anti_transpose | -1.7075094 | -1.7075094 | -1.4597837 | 0 | 0.24772567 | 0 | 0.75311661 | True | False |
| 12 | 0934a4d8:o0 / flip_ud | -3.9398098 | -3.9398098 | -3.9872878 | 0 | 0.047478041 | 0 | 0.75 | True | True |
| 24 | 0934a4d8:o0 / anti_transpose | -1.3620032 | -1.3620032 | -1.5875356 | 0 | 0.22553241 | 0 | 0.99185658 | True | False |
| 24 | 0934a4d8:o0 / flip_ud | -2.4673028 | -2.4673028 | -2.4637018 | 0 | 0.0036010227 | 0 | 0.75 | True | True |
| 48 | 0934a4d8:o0 / anti_transpose | -0.34979684 | -0.34979684 | -0.33775168 | 0 | 0.012045159 | 0 | 0.50000763 | True | True |
| 48 | 0934a4d8:o0 / flip_ud | -3.2115211 | -3.2115211 | -3.1899254 | 0 | 0.021595696 | 0 | 0.99998665 | True | True |

Root cause classification: `FULL_SEQUENCE_FORWARD_NUMERIC_PATH_DIFFERS_FROM_INCREMENTAL_KV_PATH`.
G1 allowed: `True`.
Required G1 scoring path: `INCREMENTAL_KV_REPLAY`.
Scientific configuration changed: NO.  TTT rerun: NO.  Greedy rerun: NO.  V5 rerun: NO.
