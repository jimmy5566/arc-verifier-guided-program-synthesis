# LLM Program Synthesis V2 failure forensics

This is a deterministic post-hoc analysis of frozen Pilot 50 checkpoints. It performed no inference, used no GPU, and did not load ARC grids or solutions.

## symbolic_parameter_solver

### Funnel

| Response | JSON | Schema | Macro API | Type | Parameter | Composition | Compile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 132 | 132 | 53 | 53 | 0 | 0 | 0 | 0 |

### Primary failure classes

- `PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER`: 79
- `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH`: 53
- `JSON_PARSE_FAILURE/MALFORMED_JSON`: 1
- `PROVIDER_FAILURE/CONTEXT_WINDOW_EXCEEDED`: 1

### Contract evidence

- Invalid fields: axis (31), selector (18), orientation (15), alignment (12), direction (1), color_source (1), translation (1).
- Type mismatch pairs: Objects -> Grid (38), Region -> Grid (4), Grid -> Sequence (4), Objects -> Sequence (3), Grid -> SelectedObject (1), Grid -> Pattern (1), Grid -> Objects (1), Objects -> SelectedObject (1).
- Unknown Macro IDs: none.
- Mean completion tokens: schema-valid 505.5660377358491, schema-invalid response 540.4430379746835; confirmed max-token cutoffs 0.

### Nearest to type-valid

- `3ac3eb23` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `3ac3eb23` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4cd1b7b2` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4cd1b7b2` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4e45f183` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4e45f183` candidate 2: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `5adee1b2` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `5d2a5c43` candidate 2: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `6165ea8f` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `6855a6e4` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.

## direct_parameter_ablation

### Funnel

| Response | JSON | Schema | Macro API | Type | Parameter | Composition | Compile |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 137 | 137 | 32 | 32 | 0 | 0 | 0 | 0 |

### Primary failure classes

- `PARAMETER_FAILURE/INVALID_SYMBOLIC_PARAMETER`: 105
- `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH`: 32
- `JSON_PARSE_FAILURE/MALFORMED_JSON`: 3
- `PROVIDER_FAILURE/CONTEXT_WINDOW_EXCEEDED`: 1

### Contract evidence

- Invalid fields: axis (42), orientation (24), selector (19), alignment (18), count_kind (1), color_source (1).
- Type mismatch pairs: Objects -> Grid (25), Region -> Grid (5), Objects -> SelectedObject (1), Grid -> Sequence (1).
- Unknown Macro IDs: none.
- Mean completion tokens: schema-valid 591.21875, schema-invalid response 598.7619047619048; confirmed max-token cutoffs 0.

### Nearest to type-valid

- `3eda0437` candidate 2: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4cd1b7b2` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `4e45f183` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `5adee1b2` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `5d2a5c43` candidate 2: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `6165ea8f` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `6855a6e4` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `6855a6e4` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `845d6e51` candidate 1: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.
- `846bdb03` candidate 0: `TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH` at `steps[1].macro_id` — Change the preceding macro chain or this macro so the produced concept matches this macro's declared input concept.

## Symbolic versus direct

Symbolic parameterization had higher initial frozen macro-schema throughput (53/132 versus 32/137), but neither condition produced a type-valid program. It shifted some failures to concept-chain mismatches; it did not demonstrate ParameterSolver benefit because no candidate reached that stage.

## Validator audit

- Deterministic audit sample: 26 candidates (10 schema-invalid, 10 type-invalid, plus every unusual/provider failure).
- Possible validator bugs: 0.

## Interpretation

The frozen V2 Pilot failed primarily at the LLM-to-Macro-DSL parameter-wrapper and concept-chain interface before the ParameterSolver, compiler, executor, or HardVerifier could be meaningfully evaluated.

The compiler, ParameterSolver, executor, and HardVerifier were not reached by any frozen candidate, so this pilot cannot support a claim about their effectiveness on ARC transformations.
