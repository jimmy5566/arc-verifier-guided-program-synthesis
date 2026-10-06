# Novel training data V1.1

- status: PASS_READY_FOR_GPU_BENCHMARK
- OFFICIAL_SYSTEMATICITY_PROTOCOL_READY: true
- STRICT_THREE_WAY_FAMILY_ISOLATION: false
- GPU_BENCHMARK_READY: true
- scientific claim: Validation and holdout are episode-disjoint and both evaluate composition templates unseen during training, reproducing the official upstream systematicity protocol.
- branch: training/novel-data-v1.1-scientific-split-curriculum
- systematicity upstream counts: train 82908, validation 8546, holdout 8546
- accepted episode counts: train 83607, validation 8646, holdout 8646
- family counts: train 22, validation 4, holdout 4
- 1D family split: train 14, validation 2, holdout 2
- recommended policy: FAMILY_UNIFORM_GLOBAL
- provisional novel/replay range: novel 70-80%, replay 20-30%
- dataset fingerprint: 3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82
- GPU TRAINING STARTED: false
