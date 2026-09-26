# Dual-grid learning switch — 2026-09-26

The user authorized switching the running learner to the implemented dual-grid
architecture. The old focused campaign was stopped through its STOP request;
status confirmed `stopped` and `paused_on_exit: true`.

Preserved old campaign: `live-learning-20260926-131801-321bd8`, 381 updates,
642,665 saved training steps. Verified checkpoint `real-episode-381.zip` SHA256:
`749f5e0dfc5af21963443806e1bbf192fed71195420910c484046ded838871f1`.
No old weights were migrated to the new architecture.

## Bounded real-game acceptance

Run `live-learning-20260926-201255-c867a1` used `th10-dual-grid-v1`, one play,
maximum 2,400 decisions and 300 seconds, starting from the managed paused game.
It completed 1,205 real decisions and terminated at game over with three hits.
Recorded successive decision frames all differed by exactly two game frames.
The trace included up to 77 bullets and 14 enemies. Model input keys were only
`global_grid`, `local_grid`, `player`, and `previous_rewards`.

PPO changed parameters at the game-over boundary. Save/reload preserved actions,
parameters and step count. The first new checkpoint is `real-episode-1.zip`,
SHA256 `5e50e393d5e59c8d52f7ae6c77971a32a9cb216b853b4a0a6074a9f209da5ca7`.
This validates the observation/control/update/save pipeline, not improved play
or a new instrumentation-based hitbox-boundary validation.

## Continuous campaign

Started `live-learning-20260926-201451-940156` from that new checkpoint with
`-DualGrid -Continuous -ContinueManaged -MaxSteps 18000 -NoUI`.
Confirmed status `running`, phase `playing`, `continuous: true`, new contract,
and no recorded error. The existing reward v9 coefficients were unchanged.
The dashboard service was restarted to load the new architecture filtering.

Old campaign files and checkpoints remain in place. A clean switch to the new
observation architecture necessarily starts a new policy; update counts and
growth plots should not be interpreted as continuation of the old policy.
