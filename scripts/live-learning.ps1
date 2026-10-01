param(
    [ValidateSet('rehearse', 'status', 'stop')][string]$Action = 'status',
    [ValidateRange(1, 5)][int]$Episodes = 3,
    [ValidateRange(32, 18000)][int]$MaxSteps = 1800,
    [ValidateRange(30, 900)][int]$MaxSeconds = 600,
    [switch]$ResumeLatest,
    [switch]$ContinueManaged,
    [switch]$ResumePaused,
    [switch]$Extended,
    [switch]$DualGrid,
    [switch]$DirectML,
    [switch]$DetailedLogs,
    [switch]$EvasionOnly,
    [switch]$NoUI,
    [switch]$Continuous,
    [string]$TransferEvasion,
    [switch]$UpgradeProgressPower,
    [switch]$AddPowerReward,
    [ValidateRange(0,5)][int]$ShotStudyGames = 0,
    [ValidateSet(1.0,0.5)][double]$ShotRewardScale = 1.0,
    [string]$ResumeCheckpoint
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$recordPath = Join-Path $root '.runtime/live-learning.json'
if ($Action -eq 'rehearse') {
    if ($EvasionOnly -and -not $DualGrid) { throw 'EvasionOnly requires DualGrid' }
    $profile = Get-Content (Join-Path $root 'configs/sharu-inspired-v1.json') | ConvertFrom-Json
    $policyChoice = if ($EvasionOnly) { $profile.evasion_policy_overrides } else { $profile.full_policy_overrides }
    $expectedGridContract = if ($policyChoice.cnn_architecture -eq 'narrow_action_grid') { 'th10-dual-grid-v6-action-grid-v1' } else { 'th10-dual-grid-v6' }
    $expectedReward = if ($EvasionOnly) { 'th10-evasion-death-only-v1' } else { 'th10-rewards-v20' }
    if ($AddPowerReward) {
        if ($EvasionOnly -or -not $DualGrid -or -not $ResumeCheckpoint -or $TransferEvasion -or $UpgradeProgressPower) { throw 'Power reward addition requires a full v19 checkpoint resume' }
        $expectedReward = 'th10-rewards-v19'
    }
    if ($UpgradeProgressPower) {
        if ($EvasionOnly -or -not $DualGrid -or -not $ResumeCheckpoint -or $TransferEvasion) { throw 'Power upgrade requires a full dual-grid checkpoint resume' }
        $expectedReward = 'th10-rewards-v17'
    }
    # Validate all resume contracts before replacing the active-run record or
    # stopping its observer. Python also checks hash, weights and schedule below.
    if ($ResumeLatest -or $ResumeCheckpoint) {
        $resumeStatusPath = $null
        if ($ResumeCheckpoint) {
            $resumeStatusPath = Join-Path (Split-Path $ResumeCheckpoint -Parent) 'status.json'
        } elseif (Test-Path -LiteralPath $recordPath) {
            $resumeRecord = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
            if ($resumeRecord.RunId -notmatch '^live-learning-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$') { throw 'Invalid managed run' }
            $resumeStatusPath = Join-Path $root "artifacts/$($resumeRecord.RunId)/status.json"
        }
        if ($resumeStatusPath) {
            $resumeStatus = Get-Content -LiteralPath $resumeStatusPath | ConvertFrom-Json
            $expectedContract = if ($DualGrid) { $expectedGridContract } elseif ($Extended) { 'th10-focused-bullets-v2' } else { 'th10-live-observed-v1' }
            if ($resumeStatus.contract -ne $expectedContract -or $resumeStatus.reward_version -ne $expectedReward) {
                throw 'Checkpoint uses an old observation/reward contract; start a new campaign explicitly. Existing run record was preserved.'
            }
            $profile = Get-Content (Join-Path $root 'configs/sharu-inspired-v1.json') | ConvertFrom-Json
            $expectedAlgorithm = if ($policyChoice.algorithm) { $policyChoice.algorithm } else { 'PPO' }
            $expectedCnn = if ($policyChoice.cnn_architecture) { $policyChoice.cnn_architecture } elseif ($DualGrid) { 'dual_grid' } else { 'legacy' }
            $actualAlgorithm = if ($resumeStatus.algorithm) { $resumeStatus.algorithm } else { 'PPO' }
            $actualCnn = if ($resumeStatus.cnn_architecture) { $resumeStatus.cnn_architecture } elseif ($resumeStatus.contract -like 'th10-dual-grid*') { 'dual_grid' } else { 'legacy' }
            if ($expectedAlgorithm -ne $actualAlgorithm -or $expectedCnn -ne $actualCnn) { throw 'Model architecture/algorithm changed; start a fresh campaign. Existing run record was preserved.' }
            $expectedShare = -not ($policyChoice.share_features_extractor -eq $false)
            $actualShare = if ($null -eq $resumeStatus.share_features_extractor) { $true } else { $resumeStatus.share_features_extractor }
            if ($actualShare -ne $expectedShare) { throw 'Feature sharing changed; start a fresh campaign. Existing run record was preserved.' }
            if ($resumeStatus.configured_ppo.gamma -ne 0.9995 -or $resumeStatus.discount_contract.version -ne 'th10-discount-2f-v1') {
                throw 'Checkpoint uses an old discount contract; start a new campaign explicitly. Existing run record was preserved.'
            }
        }
    }
    # Reject incompatible dual-grid resumes before changing the active run record,
    # stopping its observer, or opening UI. Never migrate focused weights silently.
    if ($DualGrid -and ($ResumeLatest -or $ResumeCheckpoint)) {
        $candidateStatus = $null
        if ($ResumeCheckpoint) {
            $candidateStatus = Join-Path (Split-Path $ResumeCheckpoint -Parent) 'status.json'
        } elseif (Test-Path -LiteralPath $recordPath) {
            $previousRun = Get-Content -LiteralPath $recordPath | ConvertFrom-Json
            if ($previousRun.RunId -notmatch '^live-learning-[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$') { throw 'Invalid managed run' }
            $candidateStatus = Join-Path $root "artifacts/$($previousRun.RunId)/status.json"
        }
        if ($candidateStatus) {
            $candidate = Get-Content -LiteralPath $candidateStatus | ConvertFrom-Json
            if ($candidate.contract -ne $expectedGridContract) {
                throw 'Dual-grid requires a fresh campaign or a dual-grid checkpoint; existing model was not changed.'
            }
        }
    }
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
        # Refresh imported reward/observation contracts for both OBS sources.
        # NoUI suppresses opening a browser, but an existing viewer still needs refresh.
        if (Test-Path -LiteralPath (Join-Path $root '.runtime/dashboard.json')) {
            & (Join-Path $PSScriptRoot 'dashboard.ps1') stop
            & (Join-Path $PSScriptRoot 'dashboard.ps1') start
        } elseif (-not $NoUI) {
            & (Join-Path $PSScriptRoot 'dashboard.ps1') start
        }
        if (-not $NoUI) {
            $dashboard = Get-Content -LiteralPath (Join-Path $root '.runtime/dashboard.json') | ConvertFrom-Json
            Start-Process "http://127.0.0.1:$($dashboard.Port)/?mode=live"
        }
        $resumeArgs = @()
        if ($TransferEvasion) {
            if ($ResumeLatest -or $ResumeCheckpoint -or $EvasionOnly -or -not $DualGrid) { throw "Transfer requires a new full dual-grid campaign" }
            $resumeArgs = @("--transfer-evasion", $TransferEvasion)
        }
        if ($ResumeCheckpoint) {
            if ($ResumeLatest) { throw 'Choose ResumeLatest or ResumeCheckpoint, not both' }
            $resumeArgs = @('--resume', $ResumeCheckpoint)
        }
        if ($ResumeLatest -and -not (Test-Path -LiteralPath $recordPath)) {
            Write-Output 'No active learning campaign; starting a new model.'
        }
        if ($ResumeLatest -and (Test-Path -LiteralPath $recordPath)) {
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
        if ($DualGrid) { $gameArgs += '--dual-grid' }
        if ($DirectML) { $gameArgs += '--directml' }
        if ($EvasionOnly) { $gameArgs += '--evasion-only' }
        if ($DetailedLogs) { $gameArgs += '--detailed-logs' }
        if ($AddPowerReward) { $gameArgs += '--add-power-reward' }
        if ($UpgradeProgressPower) { $gameArgs += '--upgrade-progress-power' }
        if ($ShotStudyGames) { $gameArgs += @('--shot-study-games', "$ShotStudyGames", '--shot-reward-scale', $ShotRewardScale.ToString([Globalization.CultureInfo]::InvariantCulture)) }
        if ($ResumePaused) { $gameArgs += '--resume-paused' }
        if ($Continuous) { $gameArgs += '--continuous' }
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
