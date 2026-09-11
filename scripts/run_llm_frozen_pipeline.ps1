# Run frozen split evaluations strictly in sequence.  It never invokes two
# Ollama jobs at once and delegates all solution access to the gated finalizer.
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$runner = Join-Path $root 'scripts\run_llm_qwen_condition.py'
$finalizer = Join-Path $root 'scripts\finalize_llm_condition.py'
$seeder = Join-Path $root 'scripts\seed_llm_full_checkpoint.py'
$aggregate = Join-Path $root 'scripts\finalize_llm_hypothesis_generator_v1_real.py'
$frozen = Join-Path $root 'configs\frozen_llm_config_v1.json'

Set-Location $root

function Wait-FrozenCheckpoint([string]$Path, [string]$Label) {
    while ($true) {
        if (Test-Path $Path) {
            try {
                $checkpoint = Get-Content $Path -Raw | ConvertFrom-Json
                $done = @($checkpoint.records.PSObject.Properties).Count
                $total = if ($checkpoint.task_count) { $checkpoint.task_count } else { '?' }
                Write-Host ("[{0}] {1}/{2}" -f $Label, $done, $total)
                if ($checkpoint.complete) { return }
            } catch {
                Write-Host "[$Label] checkpoint write in progress"
            }
        } else {
            Write-Host "[$Label] waiting for checkpoint creation"
        }
        Start-Sleep -Seconds 30
    }
}

function Invoke-FrozenCondition([string]$Config, [string]$Checkpoint, [string]$Result, [string]$Report) {
    & $python $runner --task-config $Config --frozen-config $frozen --checkpoint $Checkpoint
    if ($LASTEXITCODE -ne 0) { throw "Inference failed: $Config" }
    Wait-FrozenCheckpoint $Checkpoint (Split-Path $Checkpoint -Leaf)
    & $python $finalizer --checkpoint $Checkpoint --task-config $Config --result $Result --report $Report
    if ($LASTEXITCODE -ne 0) { throw "Finalization failed: $Config" }
}

$heldConfig = 'configs\llm_held_out_frozen_v1.json'
$heldCheckpoint = 'experiments\checkpoints\LLM_HELD_OUT_FROZEN_V1.json'
Wait-FrozenCheckpoint $heldCheckpoint 'held-out'
& $python $finalizer --checkpoint $heldCheckpoint --task-config $heldConfig --result 'experiments\results\LLM_HELD_OUT_FROZEN_V1.json' --report 'reports\llm_held_out_frozen_v1.md'
if ($LASTEXITCODE -ne 0) { throw 'Held-out finalization failed' }

Invoke-FrozenCondition 'configs\llm_challenge_like_frozen_v1.json' 'experiments\checkpoints\LLM_CHALLENGE_LIKE_FROZEN_V1.json' 'experiments\results\LLM_CHALLENGE_LIKE_FROZEN_V1.json' 'reports\llm_challenge_like_frozen_v1.md'

Write-Host 'Challenge-like stage complete. Full 1000-task continuation is intentionally disabled pending LLM_FAILURE_DIAGNOSTIC_V1.' -ForegroundColor Yellow
