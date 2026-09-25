param(
    [ValidateSet('check', 'policy-check', 'rehearse', 'live-readiness', 'start', 'status', 'logs', 'stop', 'evaluate', 'report')][string]$Action = 'status',
    [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.json$')][string]$ConfigFile = 'mock-smoke.json'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$recordPath = '.runtime\training.json'
if ($Action -in @('check', 'evaluate')) {
    $running = & docker compose --profile learning ps --status running -q training
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect training service' }
    if ($running) { throw 'Training owns the bridge; wait for it to finish or stop it first' }
}
function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker failed with exit code $LASTEXITCODE" }
}
switch ($Action) {
    'live-readiness' {
        & '.\.venv\Scripts\python.exe' -m touhou_ai.training_readiness
        if ($LASTEXITCODE -ne 0) { throw 'Readiness inspection failed' }
    }
    'rehearse' {
        $output = '/artifacts/rehearsal-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        Invoke-Docker compose --profile learning run --build --rm learner python -m touhou_ai.training_rehearsal --output $output
    }
    'policy-check' {
        $output = '/artifacts/policy-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        Invoke-Docker compose --profile learning run --build --rm learner python -m touhou_ai.policy_check --output $output
    }
    'check' { Invoke-Docker compose --profile learning run --build --rm learner }
    'start' {
        $running = & docker compose --profile learning ps --status running -q training
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect training service' }
        if ($running) { throw 'Training is already running' }
        ./scripts/bridge.ps1 status | Out-Null
        if (-not (Test-Path -LiteralPath (Join-Path 'configs' $ConfigFile))) { throw 'Configuration file not found' }
        $env:TRAIN_CONFIG = $ConfigFile
        $env:RUN_ID = 'mock-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
        Invoke-Docker compose --profile learning up --build -d training
        @{RunId=$env:RUN_ID} | ConvertTo-Json | Set-Content $recordPath
        Write-Output "Started diagnostic run: $env:RUN_ID"
    }
    'status' {
        Invoke-Docker compose --profile learning ps -a training
        if (Test-Path $recordPath) {
            $record = Get-Content $recordPath | ConvertFrom-Json
            $statusPath = Join-Path "artifacts/$($record.RunId)" 'status.json'
            if (Test-Path $statusPath) { Get-Content $statusPath }
        }
    }
    'logs' { Invoke-Docker compose --profile learning logs --tail 80 training }
    'report' {
        & '.\.venv\Scripts\python.exe' -m touhou_ai.report
        if ($LASTEXITCODE -ne 0) { throw 'Run report failed' }
    }
    'stop' { Invoke-Docker compose --profile learning stop -t 30 training }
    'evaluate' {
        $record = Get-Content $recordPath | ConvertFrom-Json
        $running = & docker compose --profile learning ps --status running -q training
        if ($running) { throw 'Stop or finish training before evaluation; bridge has one episode owner' }
        $output = '/artifacts/' + $record.RunId + '/evaluation-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
        Invoke-Docker compose --profile learning run --rm learner python -m touhou_ai.runner evaluate --model "/artifacts/$($record.RunId)/model.zip" --output $output
    }
}
