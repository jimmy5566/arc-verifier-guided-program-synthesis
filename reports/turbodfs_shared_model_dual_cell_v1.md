# TurboDFS shared-model dual-cell V1 — deployment blocked

This target-blind execution-engine gate tested a single model instance serving
two same-adapter V5 logical cells on one RTX 3090. It did not load evaluation
solutions or alter TTT, Greedy, the V5 decoder configuration, or existing cell
artifacts.

The dependency-free CPU contracts passed 11/11. The GPU serial-versus-dual
comparison used 8 non-Gold cells. Candidate-set parity was `0/8` and
frontier-floor parity was `0/8`, although termination class agreed for `8/8`.
Median shared/serial work ratios were `1.479` nodes, `1.487` forwards and
`1.479` tokens. Peak reserved VRAM was `13.352 GiB` in both modes. The measured
shared throughput was `3.734` cells/min/GPU versus serial `13.850`, a `0.270x`
speedup.

The likely execution-level mechanism is batch-sensitive decoding in this
environment: single-lane and two-lane forwards did not preserve V5 candidate
or frontier-floor outcomes. That is an observed parity failure, not an
accuracy conclusion. The shared-model scheduler was not deployed to the
active Eval60 run. Existing durable cells remain preserved; the remote raw
target-blind parity artifacts are referenced by SHA256 in the companion JSON.
