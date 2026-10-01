# Next L4 GPU telemetry requirements

For any future production-equivalent timing run, persist one-second telemetry for every L4 atomically.

## Required sample fields

`timestamp`, `gpu_id`, `gpu_util_percent`, `memory_util_percent`,
`memory_used_mb`, `memory_total_mb`, `power_draw_w`, `power_limit_w`,
`temperature_c`, `sm_clock_mhz`, `memory_clock_mhz`, and `phase`.

Allowed dual-depth phases are `MODEL_LOAD`, `TTT24_TRAIN`, `TTT24_DFS`,
`TTT48_TRAIN`, `TTT48_DFS`, `RESET_TRANSITION`, `SERIALIZE`, and `IDLE`.
For a TTT24-only run use `MODEL_LOAD`, `TTT24_TRAIN`, `TTT24_DFS`,
`SERIALIZE`, and `IDLE`.

## Required aggregation per phase and GPU

- utilization: mean, median, p10, p50, p90, p95, maximum;
- VRAM: mean and peak;
- power: mean, p90, maximum;
- temperature: mean and maximum;
- phase wall seconds.

Also report:

- `GPU_ACTIVE_FRACTION`: workload samples with utilization >10%;
- `GPU_HIGH_UTIL_FRACTION`: workload samples with utilization >=80%;
- `GPU_IDLE_FRACTION`: workload samples with utilization <=10%.

Do not substitute notebook-wide averages. Preserve run identity (source,
config, model, challenge and task/test-index hashes), worker GPU UUIDs,
model-load/barrier/heartbeat/terminal timestamps, per-task source timings,
per-output/view nodes/tokens/candidates, scheduler dispatches, failures,
deadline/finalization state, coverage and fallback counts.

The current Version 3 evidence covers only 8 tasks and 11 outputs, so a larger
representative timing cohort is required for a high-confidence production SLA.
