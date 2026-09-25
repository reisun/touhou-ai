param([ValidateSet('start', 'stop', 'status')][string]$Action = 'status')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$recordPath = Join-Path $root '.runtime/dashboard.json'
if ($Action -eq 'start') {
    if (Test-Path $recordPath) {
        $record = Get-Content $recordPath | ConvertFrom-Json
        $running = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        if ($running -and $running.StartTime.ToUniversalTime().Ticks -eq ([datetime]$record.StartTime).ToUniversalTime().Ticks) {
            Write-Output "Dashboard already running: http://127.0.0.1:$($record.Port)"
            return
        }
    }
    $port = 18767
    while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        $port++
        if ($port -gt 18777) { throw 'No free dashboard port' }
    }
    New-Item -ItemType Directory -Force '.runtime' | Out-Null
    $process = Start-Process -FilePath (Join-Path $root '.venv/Scripts/python.exe') -ArgumentList @('-u', '-m', 'touhou_ai.dashboard', '--port', $port) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput '.runtime/dashboard.log' -RedirectStandardError '.runtime/dashboard.error.log'
    @{Id=$process.Id;StartTime=$process.StartTime.ToUniversalTime().ToString('o');Port=$port} | ConvertTo-Json | Set-Content $recordPath
    for ($attempt=0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 1
            if ($health.service -eq 'touhou-observer') {
                Write-Output "Dashboard: http://127.0.0.1:$port"
                return
            }
        } catch { }
    }
    throw 'Dashboard did not start; inspect .runtime/dashboard.error.log'
}
if (-not (Test-Path $recordPath)) { Write-Output 'No managed dashboard'; return }
$record = Get-Content $recordPath | ConvertFrom-Json
if ($Action -eq 'status') {
    Invoke-RestMethod "http://127.0.0.1:$($record.Port)/api/health" -TimeoutSec 3
} else {
    $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
    if ($process) {
        if ($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) { throw 'PID reused' }
        Stop-Process -Id $process.Id
        $process.WaitForExit(5000) | Out-Null
    }
    Remove-Item -LiteralPath $recordPath
    Write-Output 'Dashboard stopped'
}
