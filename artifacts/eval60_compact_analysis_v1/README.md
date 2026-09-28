# Compact Eval60 Greedy + TurboDFS evidence

This package is a compact, Git-trackable representation of existing Eval60
artifacts for audit and CPU analysis. It contains 60 tasks and 89 test outputs.

- Greedy has 1,068 cells (3 TTT depths × 4 views) and reproduces pool oracle
  **29/89**.
- TurboDFS V5 has 1,068 scheduler DONE rows, but only 777
  have an unambiguous immutable candidate-artifact link. Its export is stored as
  `INTERIM_PARTIAL` under `snapshots/`; it is not a standard final V5 freeze.
- Gold grids are never written here. Only pre-existing exact-match booleans and
  aggregate metrics are exported. This continuation is disclosed as
  `NONBLIND_USER_AUTHORIZED_CONTINUATION`.
- Pool oracle is candidate availability. It is not final Top-2 accuracy; Top-2
  additionally needs a separately frozen selector and score evidence.
- Grid values use canonical `row1;row2` serialization, with cells comma joined,
  e.g. `0,0,1;0,1,1;2,2,2`.

Load `greedy/greedy_cells.csv` and the V5 CSVs listed in `MANIFEST.json` with
any CSV reader. Raw model files, adapters, SQLite state, token telemetry, and
large search traces remain outside Git.
