# Sequential sandbox design

Each episode represents 3--8 tool/commit decisions. A latent single fault is
sampled at one hidden step. The simulator aggregates noisy step telemetry into
an episode vector. Candidate omission, weak selector score separation, tool
warnings/latency, and environment warning/state mismatch are correlated with
their corresponding fault but intentionally overlap across sources. Neither a
one-hot source flag nor a deterministic execution-status mapping is emitted.
