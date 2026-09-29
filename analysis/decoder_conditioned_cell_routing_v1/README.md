# Frozen decoder-conditioned routing audit artifacts

Run with `py -3 scripts/run_decoder_conditioned_cell_routing_audit.py --repo .`.
All outputs are derived from existing frozen CPU-readable artifacts. `decoder_cell_labels.csv` is the authoritative joined label table. Unknown V5 provenance stays blank/unknown, not false. L1 and L2 use lower-is-better frozen train-side NLL rankings; L3 is deliberately unavailable unless production checkpoint hashes match exactly.
