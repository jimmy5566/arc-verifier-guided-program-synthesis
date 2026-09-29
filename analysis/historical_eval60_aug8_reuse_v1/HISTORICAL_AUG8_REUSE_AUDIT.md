# Historical Eval60 Aug8 Greedy reuse audit

## Result

Two actual historical Eval60 TTT24/Aug8 greedy artifacts were found. They cover all 60 current Eval60 tasks, all 28 frozen hard outputs have matching output indices, and the historical augmentation list is the exact eight-view geometry set.

Exact reuse is nevertheless blocked: neither historical artifact retains a per-task TTT24 adapter checkpoint SHA256. This fails hard identity gate 6. The compact current exports also lack enough resolved tokenizer/serialization and full greedy-semantic fields to pass every remaining gate. Therefore no historical candidate was compared to Gold, and no historical result was merged into 33/89.

## Availability

- Frozen hard cohort: 28 outputs.
- Omitted views: rot90, rot180, rot270, flip_lr.
- Expected historical source rows: 224 (two sources × 28 outputs × four views).
- Historical predictions available: 200 raw source rows; 100/112 unique logical omitted-view cells.
- Exact-reusable cells: 0.

## Scientific status

This is historical matched-control evidence only. It does not support or refute TTT24 omitted-D4 greedy coverage for the current adapters. Frozen60 is explicitly not used as an Eval60 substitute.
