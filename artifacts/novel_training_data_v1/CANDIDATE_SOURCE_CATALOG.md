# Candidate source catalog

Public-source audit frozen for novel-data-v1.

- **1d_arc** — `ACCEPTED_FOR_TRAINING` — independent 1D ARC-style benchmark with 18 named generators (https://github.com/khalil-research/1D-ARC)
- **compositional_arc** — `ACCEPTED_FOR_TRAINING` — independent visual grammar generator; raw redistribution forbidden by dataset gate (https://huggingface.co/datasets/mainlp/Compositional-ARC)
- **optozorax_arc_1d** — `REJECTED` — declared reimplementation of 1D-ARC families; not an independent novel family (https://github.com/optozorax/arc_1d)
- **reasoning_gym_arc_1d** — `REJECTED` — derived procedural implementation of 1D-ARC; generator lineage not independent (https://github.com/open-thought/reasoning-gym)
- **larc** — `REJECTED` — language annotations over official ARC tasks; source family is already seen (https://github.com/awslabs/llm-arc)
- **augarc** — `REJECTED` — augmentations of official ARC tasks and source-family seen (https://github.com/khalil-research/ARC-Aug)
- **arc_agi_1** — `REJECTED` — official ARC lineage; training overlap and evaluation governance (https://github.com/fchollet/ARC-AGI)
- **arc_tgi** — `QUARANTINE_LICENSE_UNKNOWN` — paper reports generators but an exact pinned license-clear public corpus was not established (https://arxiv.org/abs/2506.11972)
- **dbigham_arc** — `PROVENANCE_ONLY` — mixed official/custom collection; task-level independent lineage not established (https://github.com/dbigham/ARC)
- **h_arc** — `QUARANTINE_LICENSE_UNKNOWN` — official-task annotations/traces; not base-puzzle novelty (https://github.com/Le-Gris/h-arc)
- **barc** — `QUARANTINE_LICENSE_UNKNOWN` — official-task descriptions/traces; not base-puzzle novelty (https://github.com/xu3kev/BARC)
