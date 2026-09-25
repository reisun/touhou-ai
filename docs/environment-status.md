# Environment milestone: 2026-09-25

## Ready

- Bounded real-game PPO rehearsal is now implemented and ran three actual
  optimizer updates from 2,476 game action steps. See [real learning](real-learning.md).
  This is a measured, hit-only partial contract, not full-profile readiness.

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

### Native live adapter addition

- A hash/signature-checked native adapter now reads raw player/bullet/enemy/item
  state, applies game-local input and gates the update chain at exact boundaries.
- Two consecutive Normal/Reimu B stage-1 episodes were automatically started,
  stepped in two-game-frame increments through terminal death and restarted.
- Direction/focus/input-word checks and a three-second fail-neutral watchdog
  passed. The game was normally stopped after diagnostics.
- See [live adapter evidence and limitations](live-adapter.md). This is not yet
  the full Sharu observation/reward pipeline or a live Gym/HTTP training backend.

### Sharu-inspired profile addition

- Source-based profile, numerical observation/action encoder, event rewards and
  a masked-entity/grid PPO feature extractor are implemented separately from mock.
- All 18 learner-container tests passed after this addition, including the
  existing stop/resume/SIGTERM tests. Windows passed nine dependency-free tests
  with nine learner-dependent tests skipped as intended.
- `artifacts/policy-20260925-220338-68d49d` records successful synthetic policy
  construction, gradient finiteness and model save/reload. The untrained model has
  133,136 parameters; measured CPU model-only prediction mean 1.54 ms, p95 1.90 ms
  over 50 calls. This is not a complete game-loop latency measurement.
- No gameplay training steps or game inputs were performed by this diagnostic.
- Details, source provenance, provisional choices and live gates are documented
  in [the learning profile](sharu-learning-profile.md).

### Prior infrastructure evidence

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
Raw player/bullet/enemy/item extraction, native input, stage-1 frame gating and
cold episode restart are now implemented and tested. Stable entity IDs, enemy HP,
reward events, live laser geometry and cross-stage behavior remain unverified.
No GPU acceleration was configured on the AMD Radeon RX 5700 XT.

The user-requested sources now guide the initial gameplay policy. Coefficients
remain provisional and long training needs an explicit budget. `runner.py`
currently rejects a real-game backend entirely.

## Reentry

Run `scripts/verify.ps1` for complete infrastructure verification. Use
`scripts/learning.ps1 report` to inspect persisted run results. Operational
instructions are in learning.md; game connectivity instructions are in
windows-probe.md. Diagnostic training and the mock bridge are stopped at
handoff; Steam may remain running for normal use.
