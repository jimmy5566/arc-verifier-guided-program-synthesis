# GRID SEMANTIC RECOGNITION V1

Qwen3-8B 只根据 development train input/output pairs 进行 ARCSEMANTICIRV1 识别；未调用 downstream capability gate、Macro、Compiler、Executor、HardVerifier 或 program search。

## Overall

- primary_family_accuracy: **16.7%**
- core_operation_accuracy: **36.7%**
- object_role_accuracy: **9.8%**
- relation_f1: **13.3%**
- selection_criteria_accuracy: **3.3%**
- iteration_detection_accuracy: **43.3%**
- iteration_semantic_accuracy: **16.7%**
- conditional_logic_accuracy: **33.3%**
- composition_accuracy: **0.0%**
- output_construction_accuracy: **13.3%**
- semantic_parameter_accuracy: **3.3%**
- full_semantic_ir_exact: **0.0%**
- semantic_ir_success: **13.3%**
- slot_f1: **9.4%**

Recognition status: **GRID_RECOGNITION_WEAK**.

## Family-level

| Family | Tasks | Semantic success | Family correct |
| --- | ---: | ---: | ---: |
| COLOR_RECOLOR | 1 | 1 | 1 |
| CONDITIONAL_ROLE_LOGIC | 4 | 0 | 0 |
| GLOBAL_TRANSFORM | 1 | 0 | 0 |
| ITERATION_REPEAT | 15 | 0 | 0 |
| OUTPUT_CONSTRUCTION | 1 | 0 | 0 |
| RELATIONAL_TRANSFORMATION | 8 | 3 | 4 |

## Forensic strata

- iteration_repeat: {'tasks': 15, 'semantic_success': 0}
- relational: {'tasks': 8, 'semantic_success': 3}
- conditional_role: {'tasks': 4, 'semantic_success': 0}
- composition_execution_gap: {'tasks': 2, 'semantic_success': 1}
- output_construction: {'tasks': 1, 'semantic_success': 0}

`composition_execution_gap` 是上一轮 forensics 的执行表达缺口，而非独立 semantic COMPOSITION family；本独立 semantic gold 不会把 recolor/global transform 错标为 composition。

## Focused diagnostics

- iteration_repeat: {'tasks': 15, 'iteration_detection': 7, 'iteration_semantics': 0, 'conditional_detection': 5, 'semantic_success': 0, 'relation_f1': 0.18666666666666668, 'object_role_f1': 0.0}
- relational: {'tasks': 8, 'iteration_detection': 3, 'iteration_semantics': 3, 'conditional_detection': 3, 'semantic_success': 3, 'relation_f1': 0.15000000000000002, 'object_role_f1': 0.28571428571428575}
- conditional_role: {'tasks': 4, 'iteration_detection': 0, 'iteration_semantics': 0, 'conditional_detection': 0, 'semantic_success': 0, 'relation_f1': 0.0, 'object_role_f1': 0.16666666666666666}
- output_construction: {'tasks': 1, 'iteration_detection': 1, 'iteration_semantics': 0, 'conditional_detection': 0, 'semantic_success': 0, 'relation_f1': 0.0, 'object_role_f1': 0.0}

## Failure taxonomy

- CONDITION_WRONG: 1
- FAMILY_WRONG: 15
- IR_SCHEMA_FAILURE: 10
- ITERATION_MISSED: 5
- ITERATION_PARAMETER_WRONG: 8
- RELATION_WRONG: 20

Gold、task IDs、raw grids、test solutions、raw responses 与逐题 IR 均保留在 ignored private artifacts。
