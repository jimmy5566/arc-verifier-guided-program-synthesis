# A1.1 masked-Batch2 post-freeze Gold audit

Gold was opened only after `RAW_TARGET_BLIND_FREEZE.json` was verified.

- Raw freeze SHA256: `ce6afb4fc5f33901fa7bbe0813d7df7cae7f24a12f5a85960bf2298d79a1673b`
- GPU used: no
- Raw artifacts modified: no

| cell | scalar hit / rank / node | batch2 hit / rank / node | preserved |
|---|---:|---:|---|
| `97d7923e:o0:d12:identity` | False / None / None | False / None / None | NA |
| `97d7923e:o0:d12:flip_ud` | False / None / None | False / None / None | NA |
| `d59b0160:o0:d24:identity` | True / 7 / 35997 | True / 7 / 35996 | YES |
| `d59b0160:o0:d24:flip_ud` | True / 18 / 39945 | True / 19 / 40629 | YES |

KNOWN_GOLD_CANARIES = 2
BATCH2_GOLD_PRESERVED = 2/2
BATCH2_NEW_GOLD_GAINS = 0
BATCH2_GOLD_LOSSES = 0
