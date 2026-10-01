# Observation v4: per-bullet motion offsets

> Superseded by [observation v5 / rewards v16](observation-v5-rewards-v16.md),
> which implements the P layer and spell/reward changes listed below.

Implemented 2026-09-27. Contract: `th10-dual-grid-v4`.

The managed v3 campaign `live-learning-20260927-053213-83a7d2` was stopped at the
user's request before editing. Its status confirmed `stopped`, 307 completed
updates, and `budget_or_stop_before_game_over; trajectory_not_updated`.
Existing checkpoints and logs are retained. Training has not been restarted.

## Encoding

- Local: 6 x 96 x 96. Player coverage; bullet coverage at offsets 0, 2, 4 frames;
  laser coverage; validated laser coverage.
- Global: 13 x 56 x 48. Bullet density at offsets 0, 2, 4 frames; enemy density;
  known HP ratio; HP-known fraction; known maximum HP; boss density; item density;
  player location; laser coverage; validated laser coverage; player-shot coverage.
- Bullet and enemy numeric velocity grid channels have been removed.
- Each collision-enabled bullet is shifted independently by `position + frames *
  velocity_raw` before rasterization or point binning. Units are game pixels per
  frame. Never offset an averaged cell velocity. Current player position anchors
  every local layer; the viewport is not moved with the player velocity.
- Bullets outside the current viewport can enter an offset layer. Each layer is
  clipped independently. Inactive bullets are excluded from all three layers.
- Local layers retain fractional AABB coverage with maximum-overlap merging.
  Global layers retain point density `log1p(count)/log(17)`, clipped to 1.
- These are spatial encodings of currently observed motion, not guaranteed future
  game states: acceleration, curvature, future spawns and despawns are not modeled.
- Raw velocity remains available to construct the offsets. The existing numeric
  self-player state is unchanged; it is not a cell-averaged entity velocity.

The policy still uses six local channels; global channels decrease from 15 to 13.
Float16 grid storage decreases from 191232 to 180480 bytes per transition.
The additional rasterization requires CPU work despite the smaller model input.

## Compatibility and display

Both Python resume validation and the PowerShell launcher require the new
contract. A v3 model must not silently resume with the changed channel meanings.
Starting a new campaign remains a separate action; this change does not start one.
OBS shows 4F and 2F offsets with progressively stronger opacity behind current
bullets only for v4 packets. The model retains separate channels; the display is
a reconstruction, not an exact rendering of every numeric layer.

## Remaining agreed work

- P-only global channel weighted by collectible Power, excluding P from item
  density. Once implemented the global grid will have 14 channels.
- Spell breakthroughs and the planned progress reward adjustment.

Those changes are not included in this focused motion-observation revision.

## Validation

Synthetic timing report: `artifacts/dual-grid-v4-check-20260927/report.json`.
2000-bullet encoding median 10.00 ms, p95 10.43 ms; model-only inference on the
preview scene median 2.60 ms, p95 3.09 ms. This excludes native sampling, logging,
and optimizer time, and is not evidence of improved gameplay or 30 Hz end-to-end.

Tests cover opposed overlapping bullets, stationary bullets, independent viewport
entry/clipping, inactive bullets, fractional raster comparison against a scalar
implementation, removed enemy velocity, PPO update/save/reload, old-contract
rejection, and preservation of player-shot channels. OBS offset reconstruction
also has a regression test. An existing timed-bomb collector test expected the
old damage coefficient; its 1-HP/Power-0 expectation was corrected to v15's 0.015.
