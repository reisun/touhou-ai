param([Parameter(Mandatory = $true)][string]$Python)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
New-Item -ItemType Directory -Force '.runtime' | Out-Null
if (-not (Test-Path '.env')) {
    $token = & '.\.venv\Scripts\python.exe' -c 'import secrets; print(secrets.token_hex(32))'
    if ($LASTEXITCODE -ne 0) { throw 'Token generation failed' }
    Set-Content -LiteralPath '.env' -Value "BRIDGE_TOKEN=$token" -Encoding ascii
}
Write-Output 'Windows environment ready.'
