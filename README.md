# Touhou AI

Current setup and verified boundaries: [environment status](docs/environment-status.md).

Environment foundation for a Windows Touhou 10 bridge and WSL/Docker learner.
The current backend is a deterministic protocol mock, not Touhou emulation.
The mock bridge neither reads game memory nor sends keyboard input. A separate
Windows probe checks read-only game memory access. Real-game learning is not
enabled; only small, zero-reward diagnostic PPO runs are configured.

## Windows

Use PowerShell 7 and Python 3.11+:

```powershell
./scripts/setup.ps1 -Python C:\path\to\python.exe
./scripts/bridge.ps1 start
./scripts/bridge.ps1 smoke
./scripts/bridge.ps1 status
./scripts/test.ps1
./scripts/verify.ps1
./scripts/bridge.ps1 stop
```

Setup creates an isolated `.venv` and a random authentication token in `.env`.
Logs and the managed process record are in `.runtime`. Both are gitignored.
The bridge binds only to `127.0.0.1:18765`; no firewall changes are required.
Only one client may own the mock episode. Smoke testing resets that episode.

## Docker / WSL

Docker Desktop must use Linux containers with Ubuntu WSL integration enabled.
Run from the project directory:

```sh
docker compose run --build --rm test
docker compose --profile integration run --build --rm smoke
docker compose --profile learning run --build --rm learner
```

The integration check needs the Windows bridge running. Containers connect
through `host.docker.internal`, not container localhost. `.env` supplies the
token. Compose does not expose any ports or install packages into WSL.
An authentication token prevents casual local access; this is not a public API.

## Scope

See `docs/architecture.md` for the protocol and remaining real-game work.
The user decides reward design, experiment conditions, and long training runs.
The learner image includes CPU PyTorch, Gymnasium, and Stable-Baselines3 PPO.
Its default check validates the mock bridge's Gymnasium contract and needs the
Windows mock bridge running. `verify.ps1` manages that bridge, runs all tests,
executes a 64-step diagnostic run, and reloads/evaluates its saved model.
The image digest and resolved dependency versions are pinned. Installation
follows https://stable-baselines3.readthedocs.io/en/master/guide/install.html.
GPU acceleration, real-game observations/actions, frame synchronization,
and automatic episode restart remain separate implementation milestones.

## Experiment Operations

```powershell
./scripts/bridge.ps1 start
./scripts/learning.ps1 start
./scripts/learning.ps1 status
./scripts/learning.ps1 logs
./scripts/learning.ps1 evaluate
./scripts/learning.ps1 report
./scripts/learning.ps1 stop
./scripts/bridge.ps1 stop
```

The default run is a 64-step mock diagnostic, not gameplay training. Wait for
completion before evaluation. Runs persist under `artifacts/<run-id>` with
config, versions, model, checkpoints, CSV/text logs, status, and evaluation.
See [learning operations](docs/learning.md) for stopping and resuming runs.
See [Windows probe](docs/windows-probe.md) for read-only real-game diagnostics.

## Game process

`game.local.json` contains this machine's executable path and SHA256 and is
gitignored. On another machine, use `game.example.json` as the configuration
shape and record the verified executable hash.

```powershell
./scripts/game.ps1 inspect
./scripts/game.ps1 start
./scripts/game.ps1 status
./scripts/game.ps1 stop
```

Start defaults to ordinary Steam `-applaunch 1100140`, without a compatibility
override. Steam must be signed in. Select windowed mode
and skip the game's display-mode prompt for unattended startup.
If Steam is already elevated, start refuses with an explanation: exit Steam
from its menu first, then use this script. The user has removed Steam's saved
RUNASADMIN setting. Normal startup and shutdown were verified without
RunAsInvoker. The optional `-RunAsInvoker` flag remains for diagnostics and
applies only to the child process environment; it cannot change privileges of
an already running Steam.

Start waits up to 60 seconds for the actual game window and matching executable
path; use `-StartupTimeoutSeconds 120` for a longer startup. An initial Steam
login can outlast that timeout; after login inspect status before retrying.
Status reports `Managed`, `PathReadable`, and `Elevated`. Process paths use
QueryFullProcessImageName with limited query access, not module enumeration.
Stop verifies PID, creation time, and executable path, requests a normal window
close, and checks actual process exit. It never force-kills the game or Steam.

Steam startup, non-elevated game ownership, and normal shutdown were verified
on this machine. `-LaunchMode Direct` remains available for diagnostics, but
the Steam route is the tested default. No UAC or permanent compatibility
settings were changed. See docs/verification.md for the investigation history.
The Windows virtual environment currently uses the Codex bundled Python as its
base interpreter; recreate it with setup.ps1 if that runtime moves or is removed.
