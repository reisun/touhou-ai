param(
    [ValidateSet('rehearse', 'status', 'stop')][string]$Action = 'status',
    [ValidateRange(1, 5)][int]$Episodes = 3,
    [ValidateRange(32, 2400)][int]$MaxSteps = 1800,
    [ValidateRange(30, 900)][int]$MaxSeconds = 600,
    [switch]$ResumeLatest,
    [switch]$ContinueManaged,
    [switch]$ResumePaused,
    [switch]$Extended,
    [switch]$NoUI,
    [string]$ResumeCheckpoint
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$recordPath = Join-Path $root '.runtime/live-learning.json'
if ($Action -eq 'rehearse') {
    $mutex = [System.Threading.Mutex]::new($false, 'Local\TouhouAI.Learning')
    $acquired = $false
    try {
        try { $acquired = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $acquired = $true }
        if (-not $acquired) { throw 'Another real-game learner is running' }
        & (Join-Path $PSScriptRoot 'observer.ps1') stop
        if ($ContinueManaged) {
            $game = Get-Content -LiteralPath (Join-Path $root '.runtime/game.json') | ConvertFrom-Json
            $process = Get-Process -Id $game.Id -ErrorAction Stop
            if ($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$game.StartTime).ToUniversalTime().Ticks) { throw 'Managed game PID was reused' }
        }
        if (-not $NoUI) {
            & (Join-Path $PSScriptRoot 'dashboard.ps1') start
            $dashboard = Get-Content -LiteralPath (Join-Path $root '.runtime/dashboard.json') | ConvertFrom-Json
            Start-Process "http://127.0.0.1:$($dashboard.Port)/?mode=live"
        }
        $resumeArgs = @()
        if ($ResumeCheckpoint) {
            if ($ResumeLatest) { throw 'Choose ResumeLatest or ResumeCheckpoint, not both' }
            $resumeArgs = @('--resume', $ResumeCheckpoint)
        }
        if ($ResumeLatest) {
            $prior = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
            if ($prior.RunId -notmatch '^live-learning-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$') { throw 'Invalid managed run' }
            $priorOutput = Join-Path $root "artifacts/$($prior.RunId)"
            $status = Get-Content -LiteralPath (Join-Path $priorOutput 'status.json') | ConvertFrom-Json
            if ($status.episodes.Count) {
                $resumeArgs = @('--resume', (Join-Path $priorOutput $status.episodes[-1].checkpoint))
            } elseif ($status.resumed_from) {
                $resumeArgs = @('--resume', $status.resumed_from)
            } else { throw 'No completed checkpoint to resume' }
        }
        $name = 'live-learning-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        $output = Join-Path $root "artifacts/$name"
        @{RunId=$name; Output=$output; ResumeCheckpoint=$(if ($resumeArgs.Count) { $resumeArgs[1] } else { $null })} | ConvertTo-Json | Set-Content $recordPath
        $env:MPLCONFIGDIR = Join-Path $root '.runtime/matplotlib'
        $gameArgs = @()
        if ($ContinueManaged) { $gameArgs = @('--continue-managed') }
        if ($Extended) { $gameArgs += '--extended' }
        if ($ResumePaused) { $gameArgs += '--resume-paused' }
        & './.venv/Scripts/python.exe' -m touhou_ai.live_learning --output $output --episodes $Episodes --max-steps $MaxSteps --max-seconds $MaxSeconds @resumeArgs @gameArgs
        if ($LASTEXITCODE -ne 0) { throw "Real rehearsal failed; see $output/status.json" }
        Write-Output "Real rehearsal completed: $output"
    } finally {
        if ($acquired) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
} else {
    $record = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
    if ($record.RunId -notmatch '^live-learning-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$') { throw 'Invalid managed run' }
    $output = Join-Path $root "artifacts/$($record.RunId)"
    if ($Action -eq 'status') { Get-Content -LiteralPath (Join-Path $output 'status.json') }
    else {
        if (-not (Test-Path -LiteralPath $output)) { throw 'Managed run has not started' }
        New-Item -ItemType File -Path (Join-Path $output 'STOP') -Force | Out-Null
        Write-Output 'Stop requested. No next play will begin; inputs are released and the game is left open.'
    }
}
