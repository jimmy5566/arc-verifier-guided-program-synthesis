# Adaptive TTT Step 1 LOO Confidence Surface

**Mode:** CPU-only analysis of frozen LOO checkpoints. Real Eval60 test targets were not opened.

- Frozen cells: **480/480**
- Exact cells: **74**
- Parse-valid cells: **480**
- Per-task cell oracle: **8/12**

## Primary self-confidence

| Signal | Available cells | AUC exact | Spearman | Router exact tasks |
| --- | ---: | ---: | ---: | ---: |
| mean_logprob_per_token | 480 | 0.8561 | 0.4455 | 4/12 |
| sequence_logprob | 480 | 0.8682 | 0.4606 | 4/12 |
| mean_token_entropy | 480 | 0.8543 | 0.4432 | 4/12 |
| mean_token_top1_top2_margin | 480 | 0.7322 | 0.2905 | 4/12 |
| minimum_token_margin | 480 | 0.8331 | 0.4301 | 3/12 |
| ttt_loss | 384 | 0.6819 | 0.2319 | 1/12 |
| ttt_recent_loss_delta | 384 | 0.4002 | -0.1272 | 2/12 |
| generation_length | 480 | 0.2135 | -0.3598 | 2/12 |

## Coverage

| Dimension | Slot | Exact cells | Task oracle | Exclusive tasks |
| --- | --- | ---: | ---: | ---: |
| depth | 0 | 12 | 4 | 0 |
| depth | 12 | 15 | 5 | 1 |
| depth | 24 | 17 | 6 | 0 |
| depth | 48 | 16 | 5 | 1 |
| depth | 72 | 14 | 4 | 0 |
| view | identity | 10 | 3 | 0 |
| view | rot90 | 10 | 3 | 0 |
| view | rot180 | 8 | 3 | 0 |
| view | rot270 | 10 | 4 | 1 |
| view | flip_lr | 11 | 3 | 0 |
| view | flip_ud | 8 | 3 | 0 |
| view | transpose | 10 | 4 | 2 |
| view | anti_transpose | 7 | 2 | 1 |
