$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$record = Get-Content '.runtime\game.json' | ConvertFrom-Json
$game = Get-Process -Id $record.Id -ErrorAction Stop
if ($game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) { throw 'PID was reused' }
$output = 'artifacts/probes/' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6) + '.json'
& '.\.venv\Scripts\python.exe' -m touhou_ai.windows_probe --pid $game.Id --output $output
if ($LASTEXITCODE -ne 0) { throw 'Read-only game probe failed' }
Write-Output "Probe saved: $output"
