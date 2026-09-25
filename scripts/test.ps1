$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Windows tests failed' }
docker compose run --build --rm test
if ($LASTEXITCODE -ne 0) { throw 'Container tests failed' }
