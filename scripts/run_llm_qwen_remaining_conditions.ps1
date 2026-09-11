# Resume the frozen Qwen conditions serially after a known active Top-5 run.
# Each Python invocation is solution-blind and checkpoint-resumable.
param([int]$WaitForProcessId)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if ($WaitForProcessId -gt 0) {
    Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
}
& $python (Join-Path $PSScriptRoot 'run_llm_qwen_full_inference.py') --budget 5 --scheduler-hints
& $python (Join-Path $PSScriptRoot 'run_llm_qwen_full_inference.py') --budget 10
& $python (Join-Path $PSScriptRoot 'run_llm_qwen_full_inference.py') --budget 10 --scheduler-hints
