# Grid Recognition Model Capacity V1

这是冻结的 30-task development-only model condition。输入固定为 V1/A2 raw grids + current deterministic features，semantic IR、prompt semantics、scorer、temperature、top-1、seed 和无 retry 规则均固定。

| Condition | Kind | Status | Schema valid | Semantic success |
| --- | --- | --- | ---: | ---: |
| M0 Qwen3-8B | generic 8B | V1 frozen reuse | 66.7% | 4/30 (13.3%) |
| M1 stronger generic | generic | unavailable | — | — |
| M2 Qwen3-4B grids15 SFT139 | specialized ARC SFT | complete, IR-schema incompatible | 0.0% | 0/30 (0.0%) |

M1 未运行：权威 notebook metadata 中仅存在 Qwen3-8B 和单独的 4B ARC-SFT source，没有更强 generic attached model；没有联网下载或替换模型。

M2 的 30 个请求都有 completion（合计 1,309 tokens），但所有回答都无法解析为 `ARCSEMANTICIRV1`，因此 30/30 schema invalid。其 tokenizer metadata 缺少 chat template，transport 层明确复用了同一 notebook 本地 Qwen3-8B template；这不改变用户 prompt 文本、IR、decoding 或训练数据。该结果说明该 specialized grid-SFT artifact 与此 semantic-IR task 的输出接口不兼容，而不是 generic capacity 的有效对照。

结论：**MODEL_CAPACITY_UNRESOLVED**。没有 M1 stronger-generic 对照，不能依据 M2 的 specialized 负结果认定模型规模是否是主瓶颈。

所有数值为 aggregate；不含 task IDs、grids、test targets、oracle IR 或 raw response。
