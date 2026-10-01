# Eval60 TTT24-only fixed4 CPU audit V1

Retrospective development attribution only. Candidate pools and source-only D1 predictions were frozen and hashed before evaluation solutions were loaded.

## Exact replay gate

- Full fixed 4+4 D1: Top-1 20/89, Top-2 28/89, oracle 30/89.
- TTT24-only: Top-1 21/89, Top-2 24/89, oracle 24/89.
- TTT48-only: Top-1 17/89, Top-2 24/89, oracle 25/89.

## Marginal value

- Removing TTT48 loses 4 Top-2 outputs and 6 oracle outputs; it gains 0 Top-2 outputs.
- Full-pool oracle gap is 2 outputs; source-only selection loss is reported separately in the CSV artifacts.

## Runtime projection

- TTT24 accounts for 47.4% of measured TTT24+TTT48 source GPU-seconds; TTT48 accounts for 52.6%.
- TTT24-only 240-task practical P95 estimate: 7.27 h; headroom to 12 h: 4.73 h.
- Runtime classification: **SAFE**, confidence **LIMITED_SMALL_N_8_TASKS**.

## Decision

**KEEP_TTT48**

TTT48 supplies exact candidates or selected Top-2 outputs that disappear in the strict TTT24-only replay; removing it is not scientifically equivalent.

## Scientific interpretation

1. Candidate recall: removing TTT48 loses 6 exact-output candidates (6.7% of all outputs; 20.0% of full-pool oracle hits).
2. Top-2 selection: the strict TTT24-only replay loses 4 full-system Top-2 solves and gains 0; net change -4.
3. Runtime: dropping TTT48 removes 52.6% of measured source GPU-seconds, not exactly 50%; the TTT24-only 259-output median is 7.03 h.
4. GPU validation: a dedicated TTT24-only live timing run is justified only as a cost validation; this CPU audit already shows a material development-set accuracy loss, so it does not justify replacing TTT48 in production.
