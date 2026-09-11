# Data exposure audit

## Publication audit — 2026-09-11

This public release was reviewed for private/restricted solution leakage, task-ID hardcoding, solution-before-inference use, credentials, personal paths, model artifacts, competition-mounted data, checkpoints, and oversized runtime output.

- **Raw ARC/competition data and solutions:** excluded. `data/`, checkpoints, submission material, and local artifacts are ignored and not staged. The published V2 aggregate contains task identifiers and aggregate metrics only; it contains no prediction grids, raw completions, train/test solution arrays, or checkpoint traces.
- **Inference-before-solutions:** preserved. V2 declares its 50 development task IDs before inference. The finalizer requires both symbolic and direct checkpoints to be complete, frozen, have the declared IDs, and match the original frozen-config SHA before it opens any solution file for exact scoring.
- **Task hardcoding:** static source review found no task-ID-specific solver branch or answer patch. Task identifiers occur in data models, deterministic split/protocol metadata, tests, and published aggregate experiment identifiers.
- **Credentials and local paths:** `.env`, Kaggle credentials, model caches, runtime metadata, and notebook state are excluded. The environment-bound V2 model locator is not published; `configs/v2_pilot_public_protocol.json` replaces it with a portable locator description while retaining the original configuration hash as provenance.
- **Models and runtime artifacts:** model weights, safetensors, GGUF files, local virtual environments, raw logs, and generated checkpoints are excluded.

## Historical scoring controls

`LLM_FAILURE_DIAGNOSTIC_V1` static analysis reads training challenges only for development structural summaries. Its retrieval scorer opened training solutions only after all three 12-task condition checkpoints had completed and frozen predictions. Existing confirmation and held-out scores were created by gated finalizers. The challenge-like checkpoint is partial (35/149) and unscored; the full 1000-task checkpoint does not exist and continuation is disabled.

The Qwen3-8B V2 Pilot 50 completed both frozen conditions after the staggered fresh-kernel load gate. The finalizer verified the complete declared-ID sets and matching frozen-config SHA before opening a training-solution file; the resulting score is recorded in `experiments/results/LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json`.

## Outcome

No restricted data, credentials, raw model artifacts, raw checkpoint traces, or absolute personal/Kaggle mount paths are intended to be present in the committed public tree. This conclusion applies to the release snapshot, not to ignored local research artifacts.

## V2 failure-forensics follow-up

`LLM_PROGRAM_SYNTHESIS_V2_FAILURE_FORENSICS` is a deterministic, post-hoc read of the already frozen V2 aggregate checkpoints. It performs no inference, uses no GPU, and does not load ARC grids or solutions. The local candidate-level CSV remains ignored because it is derived from raw checkpoint responses. The published JSON and Markdown report contain only aggregate counts, safe task identifiers already present in the pilot protocol, sanitized validator messages, and the documented diagnostic conclusion.
