# Eval60 compact dataset quality report

## Greedy

- Expected cells: 1068
- Exported cells: 1068
- Outputs: 89
- Pool oracle: 29/89
- Missing: 0
- Duplicate primary keys: 0

## TurboDFS V5 snapshot

- Scheduler DB DONE cells: 777/1068
- Trustworthy candidate-linked cells: 777
- Ambiguous cells: 291
- Missing cells: 0
- Candidate rows exported: 1332
- Complete outputs: 58/89
- Partial outputs: 31/89
- V5 pool oracle lower bound: 27/89
- Greedy union V5 lower bound: 33/89

V5 completed scheduler cells whose candidate record cannot be linked exactly are
explicitly preserved as `AMBIGUOUS`. They are not counted as Gold misses or
used as candidate evidence. The snapshot is therefore `INTERIM_PARTIAL` even
though the scheduler records 1068 DONE cells.
