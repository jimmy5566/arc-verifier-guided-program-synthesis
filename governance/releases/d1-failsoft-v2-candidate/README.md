# D1 fail-soft V2 candidate

This is an **unpublished local release candidate**. It was built from source
commit `c4d5be2d2454babca03f07f04e9f2d73970604da`.

- `scientific_algorithm_changed = false`
- `runtime_resilience_changed = true`
- Kaggle dataset/notebook push: not performed
- GPU execution: not performed
- Competition submission: not performed

The normal-save path retains the frozen `FAST_SAVE` placeholder behavior. The
competition-rerun path removes that placeholder before invoking
`run_d1_failsoft_4gpu.py` and `build_d1_failsoft_submission.py`. Full-rerun
fallbacks are challenge-bound input copies and retain their terminal source
provenance; infrastructure identity corruption remains a hard failure.

