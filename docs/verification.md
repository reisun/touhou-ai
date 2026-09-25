# Environment verification: 2026-09-25

For the subsequent learning/operations/probe milestone, see
[environment-status.md](environment-status.md). The history below concerns
the original game lifecycle setup and UAC investigation.

## Final follow-up: ordinary Steam launch passed

The user disabled Steam's "Run this program as administrator" setting.
Read-only checks found no compatibility value for steam.exe in either HKCU
or HKLM Layers. With the current non-elevated Steam running, launching using
`game.ps1 start -RunAsInvoker:$false` succeeded: PID 33924, Managed=True,
PathReadable=True, Elevated=False. Normal stop succeeded. The default now
uses ordinary Steam launch; RunAsInvoker is optional and disabled by default.
Steam cold start after this setting change was not separately tested.
The historical RUNASADMIN notes below describe the earlier investigation,
not the current configuration.

## Latest result: Steam lifecycle passed

This result supersedes the game lifecycle limitations recorded below.

- Root cause confirmed: Steam's HKCU AppCompatFlags Layers entry has
  `~ RUNASADMIN`; Steam and its games were elevated, while the controller was not.
- An external Steam URI prompted for UAC. The user cancelled that prompt.
- RunAsInvoker on a launcher cannot lower an already running elevated Steam.
  Both Steam and direct game launches handed off to that existing Steam.
- After the user closed Steam, starting Steam with process-local RunAsInvoker
  produced a non-elevated Steam. User login was required; no credentials were
  handled by the scripts.
- After login, the queued game started as PID 14836, Elevated=False. Its path
  and start time were verified against the recorded launch, then normal stop
  succeeded.
- A fresh `game.ps1 start` launched PID 11280 through Steam. Status confirmed
  Managed=True, PathReadable=True, Elevated=False and the correct game title.
- `game.ps1 stop` succeeded, and subsequent status reported no game running.
- Final state: game stopped, Steam left running at normal privilege.
- Windowed mode and skip-next-confirmation were selected by the user.
- No global UAC, registry, game binary, or permanent compatibility settings
  were changed. The Steam RUNASADMIN entry remains, so use the project script
  to start Steam, or separately change that preference deliberately.

Correction: unavailable Get-Process.Path output did not establish that all path
queries were prohibited. QueryFullProcessImageName with limited query access
successfully retrieves the path. Elevation is now checked using the process
token explicitly. Memory access and game input remain outside this test.

- Windows PowerShell / Ubuntu WSL2 / Docker Desktop Linux engine: available.
- Windows virtual environment: created using the bundled Python interpreter.
- Six protocol/state tests: passed on Windows and Linux container.
- Windows mock bridge: authenticated health/reset/75 steps/episode end passed.
- Ubuntu WSL -> Docker container -> Windows loopback bridge: passed.
- Mock bridge stop/start/restart: passed after fixing JSON timestamp conversion.
- CPU learner: torch 2.8.0+cpu, gymnasium 1.2.0, stable-baselines3 2.7.0.
- PPO model construction and single CartPole inference: passed; no training.
- User-supplied game path and SHA256: recorded in ignored game.local.json.
- Game window title: Touhou Mountain of Faith ver 1.00a; process responding.
- User approved UAC and selected fullscreen during launch verification.

## Operational limitation

The launch process was replaced after user interaction. The resulting game
process has a visible window but its executable path is unavailable to the
current process context. Therefore game shutdown, unattended startup, memory
access, and input control are NOT verified. No elevation or compatibility
settings were changed. The game is left running for the user.

## Revalidation of game lifecycle

- Existing game PID 30860: stop script failed because no matching management
  record existed. A direct normal close request returned true, but the process
  did not exit within three seconds. Request acceptance is not proof of exit.
- After the existing process disappeared, start was run again. Initial PID:
  3008. The user observed and approved a UAC prompt.
- The startup check timed out after 60 seconds while waiting for a verified
  game window; it correctly reported failure rather than success.
- After UAC approval, th10 PID 28448 appeared with a window title but an
  unreadable executable path. It did not match the initial management record.
- The stop script was retried and failed explicitly because this replacement
  process remained. Unattended start/stop is therefore NOT working.
- Updated start waits for the actual game title and checks the executable path
  before transferring ownership to a replacement process. Status now reports
  Managed and PathReadable, and explicitly reports when no game is running.
- PowerShell syntax and status checks passed. Successful real-game start/stop
  after these changes remains blocked by the elevated process handoff.

Next implementation work must address a game-control process with appropriate
permissions, or a verified supported way to run the game without elevation.
Do not disable UAC globally or report the lifecycle scripts as operational based
only on successful mock-bridge tests.

The bridge remains a mock, independent of the real game. Its successful tests
do not validate game memory offsets or actual frame synchronization.
