# Nonblind Transfer30 LOO depth snapshot

Target-blind, interim snapshot of the 13 ranking checkpoints frozen on the RunPod worker output. No evaluation solution file was opened.

| Selection | 0 steps | 12 steps | 24 steps | 48 steps | 72 steps | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1 | 4 | 6 | 2 | 0 | 1 | 13 |
| Top-2 inclusion | 8 | 12 | 3 | 2 | 1 | 26 |

`Top-2 inclusion` counts both the first and second ranked depth for every frozen task; it is therefore 26 selections across 13 task rankings.

The ranking-file aggregate SHA256 is `d81e3f414d15044140caa4f190b2e333ecf1f8a3f163fe5c27935137c63124b65`.

This is an interim routing/selection distribution, not a real-test accuracy result. The experiment remains target-blind until all candidate artifacts are frozen and scoring begins.
