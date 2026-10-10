[CmdletBinding()]
param(
  [ValidateSet('Start','Status','Stop','Install','Uninstall')][string]$Action = 'Start',
  [string]$Root = '',
  [string]$GovernorScript,
  [int]$IdleSeconds = 15,
  [int]$PausedSeconds = 15,
  [int]$ControllerRetrySeconds = 300
)

if ([string]::IsNullOrWhiteSpace($Root)) {
  $Root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..')).Path
}
$state = Join-Path $Root '.arc2-local\orchestration\ARC2_WORKFLOW_STATE.json'
$runtime = Split-Path -Parent $state
$pidPath = Join-Path $runtime 'arc2_governor.pid'
$serviceState = Join-Path $runtime 'ARC2_GOVERNOR_SERVICE_STATE.json'
$stdout = Join-Path $runtime 'logs\arc2_governor_service.stdout.log'
$stderr = Join-Path $runtime 'logs\arc2_governor_service.stderr.log'
$taskName = 'ARC2-Governor'
if (-not $GovernorScript) { $GovernorScript = Join-Path $Root 'scripts\arc2_governor.py' }

function Get-Arc2GovernorProcess {
  if (-not (Test-Path -LiteralPath $pidPath)) { return $null }
  try {
    $governorPid = [int](Get-Content -LiteralPath $pidPath -Raw -ErrorAction Stop)
    return Get-Process -Id $governorPid -ErrorAction Stop
  } catch { return $null }
}

function Install-Arc2GovernorTask {
  $py = (Get-Command py.exe -ErrorAction Stop).Source
  $arguments = "-3 `"$GovernorScript`" --daemon --state `"$state`" --pid-file `"$pidPath`" --service-state `"$serviceState`" --idle-seconds $IdleSeconds --paused-seconds $PausedSeconds --controller-retry-seconds $ControllerRetrySeconds"
  $action = New-ScheduledTaskAction -Execute $py -Argument $arguments -WorkingDirectory $Root
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365)
  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
}

if ($Action -eq 'Status') {
  $process = Get-Arc2GovernorProcess
  [pscustomobject]@{
    running = ($null -ne $process)
    pid = if ($process) { $process.Id } else { $null }
    state_path = $state
    service_state = if (Test-Path -LiteralPath $serviceState) { Get-Content -LiteralPath $serviceState -Raw | ConvertFrom-Json } else { $null }
    scheduled_task = if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { 'REGISTERED' } else { 'NOT_REGISTERED' }
  } | ConvertTo-Json -Depth 8
  exit 0
}

if ($Action -eq 'Stop') {
  $process = Get-Arc2GovernorProcess
  if ($process) { Stop-Process -Id $process.Id -ErrorAction Stop; $process.WaitForExit() }
  Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
  exit 0
}

if ($Action -eq 'Uninstall') {
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
  exit 0
}

if (-not (Test-Path -LiteralPath $state)) { throw "ARC2 workflow state missing: $state" }
if (-not (Test-Path -LiteralPath $GovernorScript)) { throw "ARC2 Governor script missing: $GovernorScript" }
New-Item -ItemType Directory -Force (Split-Path -Parent $stdout) | Out-Null
if ($Action -eq 'Install') {
  Install-Arc2GovernorTask
  [pscustomobject]@{ status='INSTALLED'; task_name=$taskName; governor_script=$GovernorScript } | ConvertTo-Json
  exit 0
}
$existing = Get-Arc2GovernorProcess
if ($existing) {
  [pscustomobject]@{ status='ALREADY_RUNNING'; pid=$existing.Id; state=$state } | ConvertTo-Json
  exit 0
}

Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
$arguments = @('-3', $GovernorScript, '--daemon', '--state', $state,
  '--pid-file', $pidPath,
  '--service-state', $serviceState, '--idle-seconds', $IdleSeconds,
  '--paused-seconds', $PausedSeconds, '--controller-retry-seconds', $ControllerRetrySeconds)
$process = Start-Process -FilePath 'py.exe' -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
for ($attempt = 0; $attempt -lt 100; $attempt++) {
  Start-Sleep -Milliseconds 100
  $running = Get-Arc2GovernorProcess
  if ($running) {
    [pscustomobject]@{ status='STARTED'; pid=$running.Id; state=$state; governor_script=$GovernorScript } | ConvertTo-Json
    exit 0
  }
}
throw 'ARC2 Governor did not publish a live PID within 10 seconds; inspect the service stderr log.'
