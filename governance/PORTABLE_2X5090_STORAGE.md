# Portable 2x5090 storage policy

The only authoritative persistent root is `/workspace-global/arc2`. It must
reside on a verified mounted Global Volume; a directory on `/root` or
`/workspace` is never a substitute.

Seed the volume once from the validated Blackwell host:

```bash
scripts/persist_5090_portable_assets.sh \
  /workspace-global/arc2 /root/arc-model /root/arc-blackwell-env-v2 \
  /root/arc-blackwell /root/arc-runtime/active_run/5090-blackwell-serial-aa-v1
```

The seed creates `models`, `wheels`, `env`, `benchmarks`, `bootstrap`, and
`frozen-runs` in a sibling staging directory, hashes every persisted file, and
renames it only after completion. It excludes credentials, token files and
solution-named artifacts.

On a fresh two-5090 Pod, set an explicit repository URL and run:

```bash
export ARC2_REPO_URL='https://github.com/OWNER/ARC2.git'
scripts/bootstrap_5090_research_env.sh
```

The bootstrap stages the validated environment and model to `/root/arc-runtime`,
sets all cache directories there, requires exactly two `RTX 5090` / `sm_120`
devices, validates package/model/xFormers identities, and runs BF16 xFormers
attention on each GPU. It never downloads a model or rebuilds xFormers.

At run end, publish only a frozen run:

```bash
scripts/sync_arc2_run_to_persistent.sh /root/arc-runtime/active_run/RUN_ID
```

This writes `frozen-runs/.RUN_ID.staging.*`, creates `SHA256SUMS`, and atomically
renames it to `frozen-runs/RUN_ID`. The generated
`local_backup_manifest.json` describes the safe second-copy set for the local PC.
