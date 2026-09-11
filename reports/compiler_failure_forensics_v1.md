# Compiler Failure Forensics V1

- 仅重放冻结 R2 checkpoint；没有新模型调用、ARC grid 或 ARC solution。
- R2 type-valid / compile-valid / type-valid+compile-invalid: 50 / 14 / 36。
- compile-invalid 子集的重放与冻结评分一致: 36/36。

## DSL–Compiler reachability gap

- 声明 Macro: 24；静态 compiler-supported Macro: 13 (54.2%)。
- 两步 Grid→Grid type-valid / compile-valid: 40 / 6 (15.0%)。
- 三步 Grid→Grid type-valid / compile-valid: 268 / 19 (7.1%)。

## First compiler blocking error taxonomy

- UNSUPPORTED_PARAMETER_FORM: 36

## 首要错误

- symbolic parameter cannot be resolved: 19
- sequence extraction needs literal orientation and index: 8
- symbolic parameter evidence is ambiguous: 6
- object transform needs a supported resolved operation: 3

## 高频 Macro / composition

- COUNT_STRUCTURES: 19
- GENERATE_FROM_COUNT: 19
- FIND_OBJECTS: 9
- SELECT_OBJECT: 9
- TRANSFORM_OBJECT: 9
- EXTRACT_SEQUENCE: 8
- TRANSFORM_SEQUENCE: 8
- BUILD_GRID_FROM_SEQUENCE: 8
- COUNT_STRUCTURES → GENERATE_FROM_COUNT: 19
- FIND_OBJECTS → SELECT_OBJECT → TRANSFORM_OBJECT: 9
- EXTRACT_SEQUENCE → TRANSFORM_SEQUENCE → BUILD_GRID_FROM_SEQUENCE: 8

## 审计结论

- 抽样 10 条及全部 unusual/internal-error 类均已按当前 compiler 重放；没有发现 compiler bug 或 internal error。
- 主因是模型在 API-only、无训练数据语境中输出了 type-valid 但不可静态解析的 symbolic parameter form；因此下一步应限制 catalogue 与 skeleton 到 compiler-reachable static forms。
