# RTX 3090 Ampere bootstrap

`3090-ampere-env-v1` reuses the hash-verified shared CUDA 12.8 runtime and
Qwen checkpoint archives already stored in `/workspace/arc2`.  It is an
infrastructure contract only: it never changes the frozen ARC scientific
configuration and its preflight never launches inference or reads targets.

On a fresh RunPod with the Global Volume mounted at `/workspace`, use the
exact bootstrap source commit, then run one of:

```bash
ARC2_REQUIRED_GPUS=1 bash scripts/bootstrap_3090_ampere_env.sh
ARC2_REQUIRED_GPUS=2 bash scripts/bootstrap_3090_ampere_env.sh
```

The script fails closed unless every requested GPU is an RTX 3090 with compute
capability `8,6`; it verifies the persisted model, Python runtime and shared
environment archive hashes before extraction.  The restored virtualenv and
model are staged under `/root/arc-runtime-3090`, leaving `/workspace/arc2` as
the durable source of truth.  The real BF16 xFormers attention smoke must pass
before the environment reports ready.

The underlying archive is named `4090-ada-env-v1` because it is the common
CUDA 12.8 runtime archive.  Its direct reuse on RTX 3090 was verified with the
actual FA3 xFormers backend; this name does not imply an Ada-only GPU contract.
