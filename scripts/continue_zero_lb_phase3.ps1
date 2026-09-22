param(
    [Parameter(Mandatory = $true)][int]$InitialPid
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$cohort = '.\artifacts\zero_lb_local_diagnosis\evaluation12_manifest.json'
$config = '.\configs\QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json'
$challenges = '.\data\raw\arc-agi_evaluation_challenges.json'
$model = 'C:\Users\asus\Desktop\models\qwen3_4b_grids15_sft139'
$native = '.\configs\nvarc_native_846d0198'
$rootOut = '.\artifacts\zero_lb_local_diagnosis\phase3'
$orchestrationLog = Join-Path $rootOut 'orchestration.log'

function Write-Event([string]$event, [int]$augmentation = 0) {
    $line = @{ event = $event; augmentation_count = $augmentation; utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Compress
    Add-Content -LiteralPath $orchestrationLog -Value $line
    Write-Output $line
}

function Assert-Frozen([string]$path, [string]$expected) {
    if (-not (Test-Path -LiteralPath $path)) { throw "missing expected artifact: $path" }
    $data = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ($data.status -ne $expected) { throw "unexpected frozen status in $path : $($data.status)" }
}

function Invoke-BSelection([int]$count) {
    $phase = Join-Path $rootOut "aug_$count"
    $candidates = Join-Path $phase 'A_candidates_frozen.json'
    $selection = Join-Path $phase 'B_selection_frozen.json'
    Assert-Frozen $candidates 'CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING'
    if (Test-Path -LiteralPath $selection) {
        Assert-Frozen $selection 'PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING'
        Write-Event 'B_SELECTION_REUSED' $count
        return
    }
    Write-Event 'B_SELECTION_START' $count
    & $py '.\scripts\rerank_native_public_reference_selection.py' '--frozen' $candidates '--challenge-path' $challenges '--model-path' $model '--native-config-dir' $native '--output' $selection '--require-cached-evidence' *> (Join-Path $phase 'rerank.log')
    if ($LASTEXITCODE -ne 0) { throw "B selection failed for $count augmentations" }
    Assert-Frozen $selection 'PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING'
    Write-Event 'B_SELECTION_COMPLETE' $count
}

function Invoke-Generation([int]$count) {
    $phase = Join-Path $rootOut "aug_$count"
    $candidate = Join-Path $phase 'A_candidates_frozen.json'
    $checkpoint = Join-Path $phase 'checkpoints'
    if (Test-Path -LiteralPath $candidate) {
        Assert-Frozen $candidate 'CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING'
        Write-Event 'GENERATION_REUSED' $count
        return
    }
    New-Item -ItemType Directory -Force -Path $phase,$checkpoint | Out-Null
    $args = @(
        '.\scripts\run_local_native_augmentation_search.py', '--cohort', $cohort, '--config', $config,
        '--challenge-path', $challenges, '--model-path', $model, '--native-config-dir', $native,
        '--output', $candidate, '--checkpoint-dir', $checkpoint, '--resume', '--stage', 'external',
        '--external-augmentation-count', "$count", '--external-worker-count', '1',
        '--generation-micro-batch-size', '1', '--likelihood-micro-batch-size', '1', '--search-beams', '1'
    )
    Write-Event 'GENERATION_START' $count
    & $py @args *> (Join-Path $phase 'run.log')
    if ($LASTEXITCODE -ne 0) { throw "generation failed for $count augmentations" }
    Assert-Frozen $candidate 'CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING'
    Write-Event 'GENERATION_COMPLETE' $count
}

try {
    Write-Event 'ORCHESTRATION_WAIT_INITIAL' 8
    Wait-Process -Id $InitialPid -ErrorAction Stop
    $initial = Get-CimInstance Win32_Process -Filter "ProcessId=$InitialPid" -ErrorAction SilentlyContinue
    Invoke-BSelection 8
    foreach ($count in 16,32) {
        Invoke-Generation $count
        Invoke-BSelection $count
    }
    Write-Event 'PHASE3_ALL_PREDICTIONS_FROZEN'
    & $py '.\scripts\run_zero_lb_local_decisive_experiment.py' '--phase' 'finalize' *> (Join-Path $rootOut 'finalize.log')
    if ($LASTEXITCODE -ne 0) { throw 'Phase-3 scoring finalization failed' }
    & $py '.\scripts\run_zero_lb_local_decisive_experiment.py' '--phase' 'report' *> (Join-Path $rootOut 'report.log')
    if ($LASTEXITCODE -ne 0) { throw 'Final zero-LB diagnosis report failed' }
    Write-Event 'PHASE3_COMPLETE'
} catch {
    Write-Event ('PHASE3_FAILED: ' + $_.Exception.Message)
    throw
}
