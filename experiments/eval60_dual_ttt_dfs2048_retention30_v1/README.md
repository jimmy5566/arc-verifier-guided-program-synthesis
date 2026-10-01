# EVAL60_DUAL_TTT_DFS2048_RETENTION30_V1

Review-only successor to the completed DFS1024 Retention30 experiment.

The sole scientific change is `search.max_expanded_nodes: 1024 -> 2048`.
The frozen 30-output / 22-task cohort, TTT recipes, 4+4 views, scalar Regret
policy, candidate cap, timeout, lane count, model, environment, and target-blind
freeze/scoring boundary are unchanged.

The 540-second cell timeout was retained after a static audit: the slowest
DFS1024 cell was 78.574 seconds and the minimum observed throughput was 12.544
nodes/second, giving a conservative projected 2048-node time of 157.148 seconds
and 382.852 seconds of timeout headroom.

`DFS1024_*` files are immutable comparison inputs.  Gold-derived comparison
rows are consumed only by the post-freeze CPU scorer; the GPU runner receives
neither solutions nor these comparison paths.

No Kaggle dataset, notebook, or run is created by the package builder.
