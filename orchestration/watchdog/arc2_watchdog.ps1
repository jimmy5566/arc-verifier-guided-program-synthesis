param(
    [Parameter(Mandatory = $true, ParameterSetName = 'ShellCommand')][string]$SupervisorCommand,
    [Parameter(Mandatory = $true, ParameterSetName = 'Direct')][string]$SupervisorFilePath,
    [Parameter(ParameterSetName = 'Direct')][string]$SupervisorArguments = '',
    [Parameter(Mandatory = $true)][string]$StatePath,
    [int]$MaxRestarts = 3,
    [int]$PollSeconds = 2
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Save-WatchdogState([hashtable]$Value) {
    $parent = Split-Path -Parent $StatePath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$StatePath.tmp"
    $Value | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

$state = @{ schema_version = 1; restart_count = 0; status = 'STARTING'; supervisor_pid = $null; persistent_failure = $false }
while ($state.restart_count -le $MaxRestarts) {
    if ($PSCmdlet.ParameterSetName -eq 'Direct') {
        $process = Start-Process -FilePath $SupervisorFilePath -ArgumentList $SupervisorArguments -PassThru
    } else {
        $process = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-Command', $SupervisorCommand) -PassThru
    }
    $state.supervisor_pid = $process.Id
    $state.status = 'RUNNING'
    Save-WatchdogState $state
    $process.WaitForExit()
    $state.last_exit_code = $process.ExitCode
    if ($process.ExitCode -eq 0) {
        $state.status = 'STOPPED_CLEANLY'
        Save-WatchdogState $state
        exit 0
    }
    $state.restart_count += 1
    if ($state.restart_count -gt $MaxRestarts) { break }
    $state.status = 'RESTARTING'
    Save-WatchdogState $state
    Start-Sleep -Seconds ([Math]::Min(30, $PollSeconds * $state.restart_count))
}
$state.status = 'PERSISTENT_SUPERVISOR_STARTUP_FAILURE'
$state.persistent_failure = $true
Save-WatchdogState $state
exit 1
