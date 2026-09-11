# Macro Capability Registry V1

The LLM-facing catalogue is deliberately separate from the 119-primitive execution registry. The machine-readable authority is [`configs/macro_capability_registry_v1.json`](../configs/macro_capability_registry_v1.json); the prompt exposes semantics and symbolic parameter sources, never compiler primitive IDs.

| Family | Macro capabilities |
| --- | --- |
| Object | `FIND_OBJECTS`, `SELECT_OBJECT`, `TRANSFORM_OBJECT`, `COPY_OBJECT`, `ALIGN_OBJECTS` |
| Region | `FIND_ENCLOSED_REGION`, `FILL_ENCLOSED_REGION`, `EXTRACT_REGION` |
| Pattern | `FIND_PATTERN`, `COMPLETE_PATTERN`, `REPAIR_PATTERN`, `COMPLETE_SYMMETRY` |
| Graph / path | `FIND_PATH`, `TRACE_PATH`, `SERIALIZE_PATH` |
| Count / generate | `COUNT_STRUCTURES`, `GENERATE_FROM_COUNT` |
| Sequence | `EXTRACT_SEQUENCE`, `TRANSFORM_SEQUENCE`, `BUILD_GRID_FROM_SEQUENCE` |
| Structure | `SPLIT_BY_SEPARATOR`, `CONNECT_STRUCTURES` |
| Iteration | `REPEAT_UNTIL_BOUNDARY`, `PROPAGATE_PATTERN` |

There are 25 macros. Every entry in the JSON registry specifies its concept input/output, semantics, permitted parameter-source types, preconditions, valid usage, exclusions, and deterministic compiler mapping. Unsupported mappings return an explicit compiler failure; they are never approximated or converted into task-ID-specific behavior.
