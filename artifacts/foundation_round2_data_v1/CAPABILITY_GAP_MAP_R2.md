# Foundation Round 2 capability-gap map

Status: `FROZEN_RESEARCH_PRIOR`

This map uses only already-exposed Round-1 aggregate results. No individual
Round-1 holdout mistake was inspected, and the priorities are not claims about
ARC hidden tasks.

## High priority

- Parameterized movement and displacement
- Boundary-sensitive transformations
- Property-based object selection
- Multi-object spatial relations
- Conditional transformations
- Counting and comparison
- State/progression transformations
- Multi-stage construction

## Medium priority

- Reflection/mirroring robustness
- Fill and padded-fill transformations
- Object extraction and placement
- Relation-conditioned recoloring

## Low priority or currently saturated

- The already-tested Compositional-ARC OOD grow/mirror/rotation/translation templates
- Already-saturated simple pattern-copy families

## Governance reset

The Novel V1.1 holdout is `RETIRED_AFTER_ROUND1_CONFIRMATION`. Its exposed
200-example confirmation sentinel is not pristine and no V1.1 holdout episode
may enter Round-2 training or model selection. Meta-Holdout is reserved without
model access.
