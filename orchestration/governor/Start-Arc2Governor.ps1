[CmdletBinding()]
param(
  [ValidateSet('Start','Status','Stop','Install','Uninstall','Watch')][string]$Action = 'Start',
  [string]$Root = '',
  [string]$GovernorScript,
  [int]$IdleSeconds = 15,
  [int]$PausedSeconds = 15,
  [int]$ControllerRetrySeconds = 300,
  [ValidateRange(1, 3600)][int]$WatchPollSeconds = 15,
  [ValidateRange(1, 10)][int]$MaxConsecutiveFailures = 2,
  [switch]$WatchOnce,
  [switch]$DisableControllerWake
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($Root)) {
  $Root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..')).Path
}
$state = Join-Path $Root '.arc2-local\orchestration\ARC2_WORKFLOW_STATE.json'
$runtime = Split-Path -Parent $state
$pidPath = Join-Path $runtime 'arc2_governor.pid'
$serviceState = Join-Path $runtime 'ARC2_GOVERNOR_SERVICE_STATE.json'
$watchdogState = Join-Path $runtime 'ARC2_GOVERNOR_WATCHDOG_STATE.json'
$stdout = Join-Path $runtime 'logs\arc2_governor_service.stdout.log'
$stderr = Join-Path $runtime 'logs\arc2_governor_service.stderr.log'
$taskName = 'ARC2-Governor'
if (-not $GovernorScript) { $GovernorScript = Join-Path $Root 'scripts\arc2_governor.py' }
$GovernorScript = [System.IO.Path]::GetFullPath($GovernorScript)

function Save-Arc2Json([string]$Path, [hashtable]$Value) {
  $parent = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  $temporary = "$Path.$PID.$([DateTime]::UtcNow.Ticks).tmp"
  $backup = "$temporary.bak"
  try {
    [System.IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 10) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $Path) {
      [System.IO.File]::Replace($temporary, $Path, $backup)
    } else {
      [System.IO.File]::Move($temporary, $Path)
    }
  } finally {
    if (Test-Path -LiteralPath $temporary) { [System.IO.File]::Delete($temporary) }
    if (Test-Path -LiteralPath $backup) { [System.IO.File]::Delete($backup) }
  }
}

function Get-Arc2Workflow {
  try { return Get-Content -LiteralPath $state -Raw | ConvertFrom-Json } catch { return $null }
}

function Get-Arc2GovernorCommandLine([int]$GovernorPid) {
  try { return (Get-CimInstance Win32_Process -Filter "ProcessId = $GovernorPid" -ErrorAction Stop).CommandLine } catch { return $null }
}

function Test-Arc2GovernorCommandLine([string]$CommandLine) {
  if ([string]::IsNullOrWhiteSpace($CommandLine)) { return $false }
  $line = $CommandLine.ToLowerInvariant()
  return $line.Contains('arc2_governor.py') -and $line.Contains('--daemon') -and
    $line.Contains($GovernorScript.ToLowerInvariant()) -and $line.Contains($state.ToLowerInvariant())
}

function Get-Arc2GovernorProcess {
  if (-not (Test-Path -LiteralPath $pidPath)) { return $null }
  try {
    $governorPid = [int](Get-Content -LiteralPath $pidPath -Raw -ErrorAction Stop)
    $process = Get-Process -Id $governorPid -ErrorAction Stop
    if (-not (Test-Arc2GovernorCommandLine (Get-Arc2GovernorCommandLine $governorPid))) { return $null }
    return $process
  } catch { return $null }
}

function Remove-StaleGovernorPid {
  if ((Test-Path -LiteralPath $pidPath) -and $null -eq (Get-Arc2GovernorProcess)) {
    [System.IO.File]::Delete($pidPath)
  }
}

function Start-Arc2GovernorProcess {
  $existing = Get-Arc2GovernorProcess
  if ($existing) { return @{ started = $true; already_running = $true; pid = $existing.Id; error = $null } }
  Remove-StaleGovernorPid
  New-Item -ItemType Directory -Force (Split-Path -Parent $stdout) | Out-Null
  $arguments = @('-3', $GovernorScript, '--daemon', '--state', $state,
    '--pid-file', $pidPath, '--service-state', $serviceState,
    '--idle-seconds', $IdleSeconds, '--paused-seconds', $PausedSeconds,
    '--controller-retry-seconds', $ControllerRetrySeconds)
  try {
    $child = Start-Process -FilePath 'py.exe' -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -ErrorAction Stop
  } catch {
    return @{ started = $false; already_running = $false; pid = $null; error = "START_PROCESS_FAILED:$($_.Exception.Message)" }
  }
  for ($attempt = 0; $attempt -lt 100; $attempt++) {
    Start-Sleep -Milliseconds 100
    $running = Get-Arc2GovernorProcess
    if ($running) { return @{ started = $true; already_running = $false; pid = $running.Id; error = $null } }
  }
  return @{ started = $false; already_running = $false; pid = $null; error = "PID_OR_COMMANDLINE_NOT_PUBLISHED_WITHIN_10_SECONDS:child_pid=$($child.Id)" }
}

function Resolve-ControllerTarget([object]$Workflow) {
  try {
    $raw = & herdr agent list 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    $agents = (($raw | Out-String | ConvertFrom-Json).result.agents)
    $saved = if ($Workflow) { $Workflow.controller_target } else { $null }
    $selected = $agents | Where-Object { $_.pane_id -eq $saved -and $_.agent -eq 'codex' } | Select-Object -First 1
    if (-not $selected) { $selected = $agents | Where-Object { $_.name -eq 'arc-controller' -and $_.agent -eq 'codex' } | Select-Object -First 1 }
    if (-not $selected) { $selected = $agents | Where-Object { $_.focused -and $_.agent -eq 'codex' } | Select-Object -First 1 }
    if ($selected) { return $selected }
  } catch {}
  return $null
}

function Invoke-ControllerRepairWake([object]$Workflow, [string]$Failure) {
  if ($DisableControllerWake) { return 'CONTROLLER_WAKE_DISABLED' }
  $target = Resolve-ControllerTarget $Workflow
  if (-not $target) { return 'CONTROLLER_TARGET_UNRESOLVED' }
  if ($target.agent_status -eq 'working') { return 'CONTROLLER_ALREADY_ACTIVE' }
  $message = "ARC2 Governor watchdog detected $MaxConsecutiveFailures consecutive Governor start/liveness failures: $Failure. Repair the Governor process host with bounded CPU-only checks. Preserve the current workflow disposition and frozen science; do not prompt Director, launch GPU work, or alter authorizations."
  try {
    & herdr agent prompt $target.pane_id $message --timeout 30000 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { return "CONTROLLER_WAKE_SENT:$($target.pane_id)" }
    return "CONTROLLER_WAKE_FAILED_EXIT_$LASTEXITCODE"
  } catch {
    return "CONTROLLER_WAKE_FAILED:$($_.Exception.GetType().Name)"
  }
}

function Watch-Arc2Governor {
  $consecutiveFailures = 0
  $alertedFailureCount = -1
  while ($true) {
    $workflow = Get-Arc2Workflow
    $stamp = [DateTime]::UtcNow.ToString('o')
    if ($workflow -and $workflow.disposition -eq 'TERMINAL') {
      Save-Arc2Json $watchdogState @{ schema_version = 1; role = 'non-scientific governor process watchdog'; status = 'WORKFLOW_TERMINAL'; observed_at = $stamp; consecutive_failures = $consecutiveFailures; controller_notification = $null }
      return
    }
    $running = Get-Arc2GovernorProcess
    if ($running) {
      $consecutiveFailures = 0
      $alertedFailureCount = -1
      Save-Arc2Json $watchdogState @{ schema_version = 1; role = 'non-scientific governor process watchdog'; status = 'GOVERNOR_HEALTHY'; observed_at = $stamp; governor_pid = $running.Id; consecutive_failures = 0; controller_notification = $null }
    } else {
      $launch = Start-Arc2GovernorProcess
      if ($launch.started) {
        $consecutiveFailures = 0
        $alertedFailureCount = -1
        $status = if ($launch.already_running) { 'GOVERNOR_HEALTHY' } else { 'GOVERNOR_RESTARTED' }
        Save-Arc2Json $watchdogState @{ schema_version = 1; role = 'non-scientific governor process watchdog'; status = $status; observed_at = $stamp; governor_pid = $launch.pid; consecutive_failures = 0; controller_notification = $null }
      } else {
        $consecutiveFailures += 1
        $notification = $null
        if ($consecutiveFailures -ge $MaxConsecutiveFailures -and $alertedFailureCount -ne $consecutiveFailures) {
          $notification = Invoke-ControllerRepairWake $workflow $launch.error
          $alertedFailureCount = $consecutiveFailures
        }
        Save-Arc2Json $watchdogState @{ schema_version = 1; role = 'non-scientific governor process watchdog'; status = 'GOVERNOR_RESTART_FAILED'; observed_at = $stamp; consecutive_failures = $consecutiveFailures; failure = $launch.error; controller_notification = $notification }
      }
    }
    if ($WatchOnce) { return }
    Start-Sleep -Seconds $WatchPollSeconds
  }
}

function Install-Arc2GovernorTask {
  $ps = (Get-Command powershell.exe -ErrorAction Stop).Source
  $arguments = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $PSCommandPath + '" -Action Watch -Root "' + $Root + '" -GovernorScript "' + $GovernorScript + '" -IdleSeconds ' + $IdleSeconds + ' -PausedSeconds ' + $PausedSeconds + ' -ControllerRetrySeconds ' + $ControllerRetrySeconds + ' -WatchPollSeconds ' + $WatchPollSeconds + ' -MaxConsecutiveFailures ' + $MaxConsecutiveFailures
  $action = New-ScheduledTaskAction -Execute $ps -Argument $arguments -WorkingDirectory $Root
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365)
  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
  Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
}

if ($Action -eq 'Status') {
  $process = Get-Arc2GovernorProcess
  [pscustomobject]@{
    running = ($null -ne $process)
    pid = if ($process) { $process.Id } else { $null }
    state_path = $state
    service_state = if (Test-Path -LiteralPath $serviceState) { Get-Content -LiteralPath $serviceState -Raw | ConvertFrom-Json } else { $null }
    watchdog_state = if (Test-Path -LiteralPath $watchdogState) { Get-Content -LiteralPath $watchdogState -Raw | ConvertFrom-Json } else { $null }
    scheduled_task = if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { 'REGISTERED' } else { 'NOT_REGISTERED' }
  } | ConvertTo-Json -Depth 8
  exit 0
}

if ($Action -eq 'Stop') {
  Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  $process = Get-Arc2GovernorProcess
  if ($process) { Stop-Process -Id $process.Id -ErrorAction Stop; $process.WaitForExit() }
  Remove-StaleGovernorPid
  exit 0
}

if ($Action -eq 'Uninstall') {
  Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
  exit 0
}

if (-not (Test-Path -LiteralPath $state)) { throw "ARC2 workflow state missing: $state" }
if (-not (Test-Path -LiteralPath $GovernorScript)) { throw "ARC2 Governor script missing: $GovernorScript" }
if ($Action -eq 'Watch') {
  Watch-Arc2Governor
  exit 0
}
if ($Action -eq 'Install') {
  Install-Arc2GovernorTask
  [pscustomobject]@{ status='INSTALLED_AND_STARTED'; task_name=$taskName; governor_script=$GovernorScript; watchdog_state=$watchdogState } | ConvertTo-Json
  exit 0
}
$started = Start-Arc2GovernorProcess
if ($started.started) {
  [pscustomobject]@{ status=if ($started.already_running) { 'ALREADY_RUNNING' } else { 'STARTED' }; pid=$started.pid; state=$state; governor_script=$GovernorScript } | ConvertTo-Json
  exit 0
}
throw "ARC2 Governor did not publish a live verified process: $($started.error)"
