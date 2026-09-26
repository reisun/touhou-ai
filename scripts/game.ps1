param(
    [ValidateSet('inspect', 'start', 'stop', 'status')][string]$Action = 'inspect',
    [ValidateRange(5, 120)][int]$StartupTimeoutSeconds = 60,
    [ValidateSet('Steam', 'Direct')][string]$LaunchMode = 'Steam',
    [switch]$RunAsInvoker,
    [switch]$RecoveryStop
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'process-info.ps1')
$config = Get-Content (Join-Path $root 'game.local.json') | ConvertFrom-Json
$exe = Get-Item -LiteralPath $config.executable
$pidFile = Join-Path $root '.runtime\game.json'
switch ($Action) {
    'inspect' {
        $hash = (Get-FileHash -LiteralPath $exe.FullName -Algorithm SHA256).Hash
        [pscustomobject]@{Path = $exe.FullName; Bytes = $exe.Length; SHA256 = $hash; MatchesConfigured = ($hash -eq $config.sha256); FileVersion = $exe.VersionInfo.FileVersion}
    }
    'start' {
        if (Get-Process -Name 'th10' -ErrorAction SilentlyContinue) { throw 'th10 is already running' }
        foreach ($steam in @(Get-Process -Name 'steam' -ErrorAction SilentlyContinue)) {
            if ([TouhouProcessInfo]::IsElevated($steam.Id)) {
                throw 'Steam is already elevated. Exit Steam from its menu, then retry; RunAsInvoker cannot lower privileges of an existing Steam process.'
            }
        }
        $hash = (Get-FileHash -LiteralPath $exe.FullName -Algorithm SHA256).Hash
        if ($hash -ne $config.sha256) { throw 'Game executable changed; inspect and update configuration first' }
        New-Item -ItemType Directory -Force (Join-Path $root '.runtime') | Out-Null
        $launchTime = Get-Date
        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.UseShellExecute = $false
        $startInfo.WorkingDirectory = $exe.DirectoryName
        if ($LaunchMode -eq 'Steam') {
            if (-not $config.steamExecutable -or "$($config.steamAppId)" -notmatch '^\d+$') {
                throw 'Steam executable and numeric app ID must be configured'
            }
            $startInfo.FileName = $config.steamExecutable
            $startInfo.ArgumentList.Add('-applaunch')
            $startInfo.ArgumentList.Add([string]$config.steamAppId)
        } else {
            $startInfo.FileName = $exe.FullName
        }
        if ($RunAsInvoker) { $startInfo.Environment['__COMPAT_LAYER'] = 'RunAsInvoker' }
        $process = [System.Diagnostics.Process]::Start($startInfo)
        # The launch process may be Steam; only a verified game can become managed.
        @{LaunchMode = $LaunchMode; LauncherId = $process.Id; RequestedAt = $launchTime.ToUniversalTime().ToString('o'); RunAsInvoker = [bool]$RunAsInvoker} | ConvertTo-Json | Set-Content (Join-Path $root '.runtime\game-launch.json')
        Write-Output 'Waiting for the game window. Steam must be signed in; inspect status if startup times out.'
        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        do {
            $candidates = @(Get-Process -Name 'th10' -ErrorAction SilentlyContinue | Where-Object {
                $_.StartTime -ge $launchTime.AddSeconds(-1) -and
                $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -match 'ver 1\.00'
            })
            if ($candidates.Count -eq 1) {
                $candidate = $candidates[0]
                $candidatePath = [TouhouProcessInfo]::ImagePath($candidate.Id)
                if ($candidatePath -ne $exe.FullName) { throw 'Game process path differs from configured executable' }
                @{Id = $candidate.Id; StartTime = $candidate.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content $pidFile
                if ([TouhouProcessInfo]::IsElevated($candidate.Id)) {
                    throw 'Game started elevated; unattended control is not verified. See status.'
                }
                Write-Output "Game window verified: PID $($candidate.Id). Shutdown still requires a successful stop test."
                return
            }
            Start-Sleep -Milliseconds 500
        } while ((Get-Date) -lt $deadline)
        throw 'Game window was not verified before timeout. Launch record retained for diagnosis; run status.'
    }
    'status' {
        $games = @(Get-Process -Name 'th10' -ErrorAction SilentlyContinue)
        if ($games.Count -eq 0) { Write-Output 'No game running'; return }
        $record = if (Test-Path $pidFile) { Get-Content $pidFile | ConvertFrom-Json } else { $null }
        foreach ($game in $games) {
            $managed = $record -and $game.Id -eq $record.Id -and
                $game.StartTime.ToUniversalTime().Ticks -eq ([datetime]$record.StartTime).ToUniversalTime().Ticks
            $imagePath = try { [TouhouProcessInfo]::ImagePath($game.Id) } catch { $null }
            $elevated = try { [TouhouProcessInfo]::IsElevated($game.Id) } catch { $null }
            [pscustomobject]@{Id = $game.Id; Responding = $game.Responding; MainWindowTitle = $game.MainWindowTitle; Managed = [bool]$managed; PathReadable = [bool]$imagePath; Path = $imagePath; Elevated = $elevated}
        }
    }
    'stop' {
        if (-not (Test-Path $pidFile)) {
            if (Get-Process -Name 'th10' -ErrorAction SilentlyContinue) { throw 'An unmanaged game is running; close it from the game window' }
            Write-Output 'No game running'; return
        }
        $record = Get-Content $pidFile | ConvertFrom-Json
        $process = Get-Process -Id $record.Id -ErrorAction SilentlyContinue
        if ($process) {
            if ($process.StartTime.ToUniversalTime().Ticks -ne ([datetime]$record.StartTime).ToUniversalTime().Ticks) { throw 'PID was reused; refusing to close another process' }
            if ([TouhouProcessInfo]::ImagePath($process.Id) -ne $exe.FullName) { throw 'Managed process path differs from configured executable' }
            if ([TouhouProcessInfo]::IsElevated($process.Id) -and -not [TouhouProcessInfo]::IsElevated($PID)) {
                throw 'Game is elevated but this controller is not. Close the game manually and launch with a verified non-elevated route.'
            }
            $requested = $process.CloseMainWindow()
            if (-not $requested -and -not $RecoveryStop) { throw 'No closable game window; close it manually' }
            if (-not $process.WaitForExit(10000)) {
                if (-not $RecoveryStop) { throw 'Game has not exited; no forced termination performed' }
                # Kill through the verified process handle, never a name or a fresh PID lookup.
                $process.Kill()
                if (-not $process.WaitForExit(10000)) { throw 'Managed game did not exit during recovery' }
            }
        }
        $remaining = @(Get-Process -Name 'th10' -ErrorAction SilentlyContinue)
        if ($remaining.Count -gt 0) { throw 'A different game process remains, possibly after UAC/Steam handoff. Close it from its window; automatic shutdown is not verified.' }
        Remove-Item -LiteralPath $pidFile
        Write-Output 'Game stopped'
    }
}
