# Same-parent expanded-sibling separability v1

Leakage guard for the global node-prefix classifier.

- choice groups: 329
- unique tasks: 31
- rows: 674
- OOF ROC-AUC: 0.7609
- minimum fold ROC-AUC: 0.6137
- learned top-1: 0.7295
- best frozen baseline top-1: 0.7325
- top-1 gain: -0.0030
- learned MRR: 0.8622
- best frozen baseline MRR: 0.8637
- MRR gain: -0.0015

Decision: **EXPANDED_SIBLING_SIGNAL_NOT_ESTABLISHED**

This result does not cover retained-but-never-expanded siblings because the historical frozen archive did not preserve their diagnostic feature vectors.
