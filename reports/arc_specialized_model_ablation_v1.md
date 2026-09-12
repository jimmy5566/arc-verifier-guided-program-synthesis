# ARC SPECIALIZED MODEL AND INFERENCE ABLATION V1

本实验复用 Direct V1 的同一冻结 30-task development cohort。D1 完成；D2 因没有可验证的 native ARC serialization 而按 stop condition 停止，依赖 D2 的 D3--D5 没有运行。

## D0 与 D1

- D0 Any-of-2 exact：3.3%
- D1 Any-of-2 exact：0.0%
- D1 Δ：-3.3 pp，结论：**ARC_SFT_WEIGHTS_WEAK_ALONE**
- D1 valid grid rate：0.0%
- D1 wall runtime：154.7s

## 停止条件

D2：`NOT_RUN_NATIVE_INTERFACE_UNRESOLVED`。已审计 checkpoint package 与冻结本地代码；未发现其 native ARC prompt/serialization。为避免把猜测当作 NVARC recipe，没有运行 D2，也没有运行依赖其 native interface 的 D3、D4、D5。

## D0→D1 transitions

- fail_to_success: 0
- success_to_success: 0
- success_to_fail: 1
- both_fail: 29

## Strata

- Iteration: {'D0': '0/15', 'D1': '0/15', 'D2': 'NOT_RUN_NATIVE_INTERFACE_UNRESOLVED', 'D3': 'NOT_RUN_DEPENDS_ON_D2', 'D4': 'NOT_RUN_DEPENDS_ON_D3', 'D5': 'NOT_APPLICABLE_NO_D4_POOL'}
- Relational: {'D0': '0/8', 'D1': '0/8', 'D2': 'NOT_RUN_NATIVE_INTERFACE_UNRESOLVED', 'D3': 'NOT_RUN_DEPENDS_ON_D2', 'D4': 'NOT_RUN_DEPENDS_ON_D3', 'D5': 'NOT_APPLICABLE_NO_D4_POOL'}
- Conditional/Role: {'D0': '1/4', 'D1': '0/4', 'D2': 'NOT_RUN_NATIVE_INTERFACE_UNRESOLVED', 'D3': 'NOT_RUN_DEPENDS_ON_D2', 'D4': 'NOT_RUN_DEPENDS_ON_D3', 'D5': 'NOT_APPLICABLE_NO_D4_POOL'}
- Composition: {'D0': '0/2', 'D1': '0/2', 'D2': 'NOT_RUN_NATIVE_INTERFACE_UNRESOLVED', 'D3': 'NOT_RUN_DEPENDS_ON_D2', 'D4': 'NOT_RUN_DEPENDS_ON_D3', 'D5': 'NOT_APPLICABLE_NO_D4_POOL'}
- Output Construction: {'D0': '0/1', 'D1': '0/1', 'D2': 'NOT_RUN_NATIVE_INTERFACE_UNRESOLVED', 'D3': 'NOT_RUN_DEPENDS_ON_D2', 'D4': 'NOT_RUN_DEPENDS_ON_D3', 'D5': 'NOT_APPLICABLE_NO_D4_POOL'}

## Leakage audit

- D0/D1 artifacts were validated and SHA256-hashed before development targets were opened.
- D1 inference was offline and uses only train pairs plus test inputs.
- No Semantic IR, DSL, capability registry, compiler, executor, verifier, target, oracle, public, held-out, or challenge dependency occurs in D1 inference.
