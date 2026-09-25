$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$running = & docker compose --profile learning ps --status running -q training
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect training service' }
if ($running) { throw 'Training owns the bridge; stop it before verification' }
./scripts/test.ps1
$startedBridge = $false
try {
    try { ./scripts/bridge.ps1 status | Out-Null }
    catch { ./scripts/bridge.ps1 start; $startedBridge = $true }
    docker compose --profile learning run --build --rm learner python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Learner tests failed' }
    ./scripts/learning.ps1 check
    $run = 'verify-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
    docker compose --profile learning run --rm learner python -m touhou_ai.runner train --config /configs/mock-smoke.json --output "/artifacts/$run"
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostic training failed' }
    docker compose --profile learning run --rm learner python -m touhou_ai.runner evaluate --model "/artifacts/$run/model.zip" --output "/artifacts/$run/reloaded-evaluation.json"
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint reload evaluation failed' }
    Write-Output "PASS: environment workflow; artifacts/$run"
} finally {
    if ($startedBridge) { ./scripts/bridge.ps1 stop }
}
