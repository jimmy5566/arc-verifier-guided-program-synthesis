$marker = 'orchestration\smoke\watchdog_crash_once.marker'
if (Test-Path -LiteralPath $marker) { exit 0 }
New-Item -ItemType File -Path $marker -Force | Out-Null
exit 7
