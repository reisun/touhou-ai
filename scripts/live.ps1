param(
    [ValidateSet('probe', 'restart', 'verify')][string]$Action = 'probe',
    [ValidateRange(1, 1800)][int]$Steps = 150,
    [ValidateRange(1, 3)][int]$Episodes = 1
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = '.\.venv\Scripts\python.exe'
if ($Action -eq 'probe' -and $Episodes -ne 1) { throw 'Probe cannot restart episodes' }
for ($episode = 0; $episode -lt $Episodes; $episode++) {
if ($Action -in @('restart', 'verify')) {
    if (Test-Path '.runtime/game.json') {
        $prior = Get-Content '.runtime/game.json' | ConvertFrom-Json
        $mutex = $null
        if ([System.Threading.Mutex]::TryOpenExisting("Local\TouhouAI.Live.$($prior.Id)", [ref]$mutex)) {
            $mutex.Dispose()
            throw 'Another live controller owns the game; stop that controller before restart'
        }
    }
    if (Test-Path '.runtime/game.json') { ./scripts/game.ps1 stop }
    ./scripts/game.ps1 start
}
$record = Get-Content '.runtime/game.json' | ConvertFrom-Json
$game = Get-Process -Id $record.Id -ErrorAction Stop
if ($game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) { throw 'PID reused' }
$output = 'artifacts/live/' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6) + '.jsonl'
$arguments = @('-m', 'touhou_ai.live_runtime', '--pid', $game.Id, '--steps', $Steps, '--output', $output)
if ($Action -in @('restart', 'verify')) { $arguments += '--start' }
if ($Action -eq 'verify') { $arguments += @('--movement-test', '--watchdog-test') }
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Live diagnostic failed; inspect $output" }
    Write-Output "Live diagnostic saved: $output"
} finally {
    if ($Action -eq 'verify') { ./scripts/game.ps1 stop }
}
}
