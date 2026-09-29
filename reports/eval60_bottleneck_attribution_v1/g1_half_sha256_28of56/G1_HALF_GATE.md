# G1 half stop gate

NONBLIND_MECHANISM_DIAGNOSTIC

A = #(any current V5-searchable) = 0
B = #(pruning-policy blocked) = 28
C = #(all current states probability-limited) = 0
Strict-searchable outputs = 0

Decision: STOP_G1_AND_RUN_G2
Rule: complete remaining G1 only when A > 0; otherwise jump to G2.
