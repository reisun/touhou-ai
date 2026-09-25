param([ValidateSet('start', 'stop', 'status', 'smoke')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$pidFile = Join-Path $root '.runtime\bridge.json'
if (-not (Test-Path '.env')) { throw 'Run setup.ps1 first' }
$env:BRIDGE_TOKEN = (Get-Content '.env' | Where-Object { $_ -like 'BRIDGE_TOKEN=*' }).Substring(13)
$env:BRIDGE_URL = 'http://127.0.0.1:18765'
$headers = @{Authorization = "Bearer $env:BRIDGE_TOKEN"}
switch ($Action) {
    'start' {
        if (Get-NetTCPConnection -LocalPort 18765 -State Listen -ErrorAction SilentlyContinue) {
            throw 'Port 18765 is already in use. Run status or stop first.'
        }
        $process = Start-Process -FilePath $python -ArgumentList @('-u', 'bridge.py') -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput '.runtime\bridge.log' -RedirectStandardError '.runtime\bridge.error.log'
        @{Id = $process.Id; StartTime = $process.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content $pidFile
        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 250
            try {
                Invoke-RestMethod "$env:BRIDGE_URL/health" -Headers $headers -TimeoutSec 1
                return
            } catch { }
        }
        throw 'Bridge did not become healthy; inspect .runtime logs'
    }
    'stop' {
        if (-not (Test-Path $pidFile)) { Write-Output 'No managed bridge'; return }
        $record = Get-Content $pidFile | ConvertFrom-Json
        $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        if ($process) {
            if ($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) {
                throw 'PID was reused; refusing to stop another process'
            }
            Invoke-RestMethod "$env:BRIDGE_URL/shutdown" -Method Post -Headers $headers -TimeoutSec 3 | Out-Null
            if (-not $process.WaitForExit(10000)) { throw 'Bridge did not exit after shutdown request' }
        }
        Remove-Item -LiteralPath $pidFile
        Write-Output 'Bridge stopped'
    }
    'status' { Invoke-RestMethod "$env:BRIDGE_URL/health" -Headers $headers -TimeoutSec 3 }
    'smoke' {
        & $python smoke.py
        if ($LASTEXITCODE -ne 0) { throw 'Smoke test failed' }
    }
}
