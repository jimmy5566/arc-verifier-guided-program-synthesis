# RTX 4090 Ada bootstrap

This is a runtime-only environment contract. It preserves the frozen ARC
scientific configuration and stages all active files onto local NVMe.

On a fresh RunPod Pod with the persistent volume mounted at `/workspace`:

    git clone https://github.com/jimmy5566/arc-verifier-guided-program-synthesis.git /root/bootstrap-source
    cd /root/bootstrap-source
    git checkout 7ad9c78026a6e6509fc9746075cc6e72c006a180
    ARC2_REQUIRED_GPUS=1 bash scripts/bootstrap_4090_ada_env.sh

For a two-card Ada Pod, replace the last line with:

    ARC2_REQUIRED_GPUS=2 bash scripts/bootstrap_4090_ada_env.sh

The first checkout is the immutable bootstrap implementation. The script then
checks out the separate frozen scientific source commit
`9602b851dea2b4b4ae69b25585a9e591af1de523` under `/root/arc-runtime/arc2`.
It verifies archive and model hashes, restores the pinned Python runtime and
environment locally under `/root/arc-runtime`, then runs a BF16 xFormers
fast-path smoke. It does not launch ARC inference. The two-card mode only
declares two independent task-worker bindings; it does not use model or tensor
parallelism.
