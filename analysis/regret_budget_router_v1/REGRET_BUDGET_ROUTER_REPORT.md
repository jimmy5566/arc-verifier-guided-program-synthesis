# Regret budget router v1

CPU-only, post-freeze nonblind development analysis. Historical union remains 33/89 unchanged.

- Router rows at node 1024: 13; early-terminated D2 validation cells excluded: 3.
- Unique outputs: 5; Y_CONTINUE positive/negative: 4/9; positive unique outputs: 2.
- Best 100%-recall LOO operating point: LOGISTIC_C=10; continuation=0.615; node saving=7180 (24.2%); false stops=0.
- Signal: ROUTER_SIGNAL_NOT_ESTABLISHED. This is exploratory because positives <8 and unique outputs <20.
- Top descriptive feature candidates: candidate_yield_last256, retained_successors_last256, nodes_since_last_candidate, frontier_growth_last256, frontier_size_1024.
- 4-fold SHA256 grouping is emitted as an additional exploratory diagnostic; LOO is primary.
- Pause/resume current implementation: NO / NOT_RUN; see static audit.
- Expansion plan: 60 rows, estimated 3.17 GPU-hours.
