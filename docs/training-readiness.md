# Live training readiness

Status: the FULL profile is not ready. A bounded real-game PPO collector now ran
three verified optimizer updates. See [real-learning.md](real-learning.md) for
the actual evidence, partial contract, commands and remaining full-profile gates.
The synthetic command below remains a separate unit/integration diagnostic.

## Available commands

```powershell
./scripts/learning.ps1 live-readiness
./scripts/learning.ps1 rehearse
```

The rehearsal runs the numerical observation encoder, the same PPO feature
extractor and action heads, and the event reward calculator in an explicitly
synthetic, action-dependent environment. It verifies finite parameter changes,
checkpoint save/load, deterministic action agreement after reload and continued
training with preserved timestep accounting. Outputs are isolated under
`artifacts/rehearsal-*`; synthetic models are named `synthetic-*.zip`.

The default is 64 training steps plus 32 resumed steps. Rollout length 32,
batch size 16 and 2 epochs are test overrides, not production settings. Each
synthetic episode is time-truncated at 32 steps. This does not validate the
planned game-over-only production update boundary or game physics. The process
is bounded and exits automatically. It neither launches nor controls Touhou.

`learning.ps1 start` remains the existing MOCK learner command. It must not be
mistaken for a real-game launch. `live-learning.ps1 rehearse` is the bounded real
collector; it does not bypass validation of missing full-profile features.

## Remaining acceptance work

1. Calibrate movement bounds, raw velocity units, collision masks, entity
   lifetime tracking and acceleration. Do not fill missing measurements with zero.
2. Validate enemy/boss HP and phase identity against observed shooting. Generate
   confirmed hit, bomb, damage, kill and stage-clear events with stable IDs.
3. Validate laser geometry and multi-stage transitions. Separate time truncation,
   game-over termination and acquisition failures.
4. Connect a single-owner live Gym collector to PPO with bounded episodes/time,
   manual stop, checkpointing and game-over-only update boundaries. Release game
   input before optimizer work; its duration may exceed the 3-second watchdog.
5. Run a bounded end-to-end real-game rehearsal, including stop, restart, crash
   cleanup, checkpoint resume and live UI reward/learner telemetry.

A reduced avoidance-only profile could be implemented separately, but it changes
the requested reward/observation policy and must be explicitly selected. It must
not be represented as completion of the full Sharu-inspired profile.
