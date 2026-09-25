# Environment milestone: 2026-09-25

## Ready

- Windows owns D:\repos\touhou-ai and the non-elevated Steam game lifecycle.
- Ubuntu WSL2 and Docker Desktop execute the same Compose project.
- Windows mock bridge exposes authenticated protocol v1 on loopback port 18765.
- The learner container connects through host.docker.internal.
- CPU PyTorch 2.8.0, Gymnasium 1.2.0, and SB3 2.7.0 are installed; image digest
  and resolved dependency versions are recorded in the Dockerfile and lock file.
- Gymnasium adapter validates observations, backend, frame advancement and reset.
- PPO runner saves unique runs, checkpoints, configuration, versions, metrics,
  status, deterministic evaluation and reloadable models.
- Managed Compose training supports status, logs, cooperative stop, evaluation,
  and reports. Resume creates a new run and checks core configuration compatibility.
- Windows read-only probe checks executable hash, process path, PE32 header and
  VM_READ access without changing game memory or issuing game inputs.

## Evidence

- Twelve tests passed inside the learner container through Ubuntu WSL.
- Host and minimal container passed six dependency-free tests; six learner-only
  tests are intentionally skipped there and run in the learner image.
- Tests include actual SIGTERM model saving, STOP handling, resumed optimization,
  overwrite refusal, invalid configs and conflicting external resets.
- Gymnasium/SB3 contract check passed through WSL -> container -> Windows bridge.
- artifacts/verify-20260925-214454-afd1cf completed 64 diagnostic timesteps and
  reloaded its model for two 600-frame mock episodes.
- artifacts/mock-20260925-214714-662fc4 completed as a background Compose service
  with exit code 0; its saved model was evaluated through the operations script.
- artifacts/probes/20260925-214126-a5d1a7.json records successful Windows VM_READ.
  The game was at the title screen; all sampled raw globals were zero.

## Deliberately not established

No real-game learning was run. The mock reward is always zero and is exclusively
an infrastructure fixture, not a selected gameplay reward or a useful policy.
Actual player/bullet/laser observations, input handling, frame synchronization,
death detection and episode restart remain to be implemented and validated.
No GPU acceleration was configured on the AMD Radeon RX 5700 XT.

The user retains decisions on gameplay reward, experiment settings, and long
training runs. `runner.py` currently rejects a real-game backend entirely.

## Reentry

Run `scripts/verify.ps1` for complete infrastructure verification. Use
`scripts/learning.ps1 report` to inspect persisted run results. Operational
instructions are in learning.md; game connectivity instructions are in
windows-probe.md. Diagnostic training and the mock bridge are stopped at
handoff; Steam may remain running for normal use.
