# Visible CMD/PowerShell progress display for the resumable Qwen ARC run.
# It does not duplicate an already-active inference runner.
param([int]$RefreshSeconds = 10)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$checkpointDir = Join-Path $root 'experiments\checkpoints'
$activeManifest = Join-Path $checkpointDir 'LLM_ACTIVE_RUN.json'
$fallbackCheckpoint = Join-Path $checkpointDir 'LLM_CONFIRMATION_DEV_100.json'
$python = Join-Path $root '.venv\Scripts\python.exe'

Set-Location $root
Write-Host 'ARC2 Qwen LLM evaluation progress monitor' -ForegroundColor Cyan
Write-Host 'This window is safe to close: it only displays progress.' -ForegroundColor Yellow
Write-Host 'The resumable inference worker keeps its own checkpoints.' -ForegroundColor Yellow

while ($true) {
    Clear-Host
    Write-Host 'ARC2 — LLM_HYPOTHESIS_GENERATOR_V1' -ForegroundColor Cyan
    Write-Host (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    Write-Host ''
    try {
        $manifest = if (Test-Path $activeManifest) { Get-Content $activeManifest -Raw | ConvertFrom-Json } else { $null }
        $activeCheckpoint = if ($manifest) { $manifest.checkpoint_path } else { $fallbackCheckpoint }
        $active = Get-Content $activeCheckpoint -Raw | ConvertFrom-Json
        $done = @($active.records.PSObject.Properties).Count
        $total = if ($active.task_count) { [int]$active.task_count } else { 100 }
        $pct = [math]::Round(100 * $done / $total, 1)
        $latest = @($active.records.PSObject.Properties | Select-Object -Last 1)[0]
        $latestText = if ($latest) { " | latest: $($latest.Name)" } else { '' }
        $status = if ($active.complete) { 'COMPLETE' } elseif ($active.paused_reason) { 'PAUSED' } else { 'ACTIVE / RESUMABLE' }
        $condition = if ($manifest) { $manifest.condition } else { 'LLM_CONFIRMATION_DEV_100' }
        Write-Host ("CURRENT {0}: {1}/{2} ({3}%)  {4}{5}" -f $condition, $done, $total, $pct, $status, $latestText) -ForegroundColor Green
        if ($active.paused_reason) { Write-Host ("Pause reason: {0}" -f $active.paused_reason) -ForegroundColor Yellow }
    } catch {
        Write-Host 'CURRENT CONFIRMATION: checkpoint write in progress' -ForegroundColor DarkYellow
    }
    Write-Host ''
    Write-Host 'Historical pilot checkpoint (not current progress):' -ForegroundColor DarkGray
    $files = Get-ChildItem $checkpointDir -Filter 'LLM_HYPOTHESIS_GENERATOR_V1_*.json' -ErrorAction SilentlyContinue | Sort-Object Name
    if ($files) {
        foreach ($file in $files) {
            try {
                $state = Get-Content $file.FullName -Raw | ConvertFrom-Json
                $done = @($state.records.PSObject.Properties).Count
                $total = if ($state.task_count) { [int]$state.task_count } else { 1000 }
                $pct = if ($total) { [math]::Round(100 * $done / $total, 1) } else { 0 }
                $status = if ($state.complete) { 'COMPLETE' } else { 'STOPPED PILOT' }
                Write-Host ("{0,-62} {1,4}/{2} ({3,5}%)  {4}" -f $file.BaseName, $done, $total, $pct, $status)
            } catch {
                Write-Host ("{0}: checkpoint write in progress" -f $file.BaseName) -ForegroundColor DarkYellow
            }
        }
    } else {
        Write-Host 'Waiting for inference checkpoint...' -ForegroundColor Yellow
    }
    Write-Host ''
    $worker = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'run_llm_qwen_(full_inference|condition)\.py' }
    $queue = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'run_llm_qwen_remaining_conditions\.ps1' }
    Write-Host ("Inference worker: {0} | queued conditions: {1}" -f ($(if ($worker) { 'ACTIVE' } else { 'not detected' }), $(if ($queue) { 'ACTIVE' } else { 'not detected' })))
    $gpu = & nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host ("GPU: {0}" -f $gpu) }
    Write-Host ''
    Write-Host 'Close this window to stop monitoring only; the inference worker continues.' -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
