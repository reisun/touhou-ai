param([ValidateSet('start', 'stop', 'status')][string]$Action='status')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location $root
$path=Join-Path $root '.runtime/observer.json'
if (Test-Path $path) {
    $record=Get-Content -LiteralPath $path | ConvertFrom-Json
    $running=Get-Process -Id $record.Id -ErrorAction SilentlyContinue
    if ($running -and $running.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) { $running=$null }
}
if ($Action -eq 'stop') {
    if ($running) {
        New-Item -ItemType File -Path (Join-Path $record.Output 'STOP') -Force | Out-Null
        if (-not $running.WaitForExit(10000)) { throw 'Observer did not stop' }
    }
    Write-Output 'Observer stopped. No game input was sent.'
    return
}
if ($Action -eq 'status') {
    [pscustomobject]@{Running=[bool]$running;Record=$record}
    return
}
if ($running) { Write-Output 'Observer already running'; return }
$game=Get-Content -LiteralPath (Join-Path $root '.runtime/game.json') | ConvertFrom-Json
$process=Get-Process -Id $game.Id -ErrorAction Stop
if ($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$game.StartTime).ToUniversalTime().Ticks) { throw 'Managed game PID changed' }
$name='observer-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
$output=Join-Path $root "artifacts/$name"
$worker=Start-Process -FilePath (Join-Path $root '.venv/Scripts/python.exe') -ArgumentList @('-u','-m','touhou_ai.observe_live','--model-monitor','--pid',$game.Id,'--output',$output) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput '.runtime/observer.log' -RedirectStandardError '.runtime/observer.error.log'
@{Id=$worker.Id;StartTime=$worker.StartTime.ToUniversalTime().ToString('o');Output=$output} | ConvertTo-Json | Set-Content $path
Write-Output "Passive observer started: $name (30 Hz target, 15-minute budget)"
