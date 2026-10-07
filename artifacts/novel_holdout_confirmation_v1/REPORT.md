# Novel Holdout Confirmation V1

| Model | Overall exact | Macro-family exact | Valid | 1D exact | Compositional exact |
|---|---:|---:|---:|---:|---:|
| SFT139 Base | 0.8000 | 0.8000 | 1.0000 | 0.6100 | 0.9900 |
| 2M LoRA | 0.8700 | 0.8700 | 1.0000 | 0.7400 | 1.0000 |

Overall absolute gain: 7.00 pp

1D absolute gain: 13.00 pp

Compositional absolute gain: 1.00 pp

Paired n01=17, n10=3; McNemar exact p=0.002576828; bootstrap 95% CI=[0.030000000000000002, 0.11].

Classification: `TRANSFER_CONFIRMED`

NOVEL_V1_1_HOLDOUT_CONFIRMATION_SENTINEL = `EXPOSED_AFTER_CONFIRMATION`
