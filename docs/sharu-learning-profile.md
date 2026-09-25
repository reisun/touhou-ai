# Sharu-inspired initial learning profile

Checked 2026-09-25. This is an independently implemented starting point, not
Sharu Studio's original code, model, or a claim of identical learning behavior.

## Sources and interpretation

- Primary source: the description of [the requested live video, episode #23](https://www.youtube.com/watch?v=t2rOpTryNU4).
  The description was retrieved as public YouTube metadata; the entire stream was
  not watched. Its September 24 changes supersede older episode descriptions.
- [SharuStudio's public X profile](https://x.com/SharuStudio) was checked without
  signing in. Only publicly visible recent/pinned posts were accessible, not the
  entire history. The profile also discusses narration, which is separate from
  the gameplay learner and is not included here.
- [SB3 2.7 custom policy documentation](https://stable-baselines3.readthedocs.io/en/v2.7.0/guide/custom_policy.html)
  informs the local MultiInputPolicy/feature-extractor implementation.

The source describes PPO, learning without demonstrations, Normal/Reimu B,
numerical game state, a single real-time game, and 30 decisions per second.
Selected observations are 128 nearby bullets (including motion, geometry and
predicted approach), a 4-pixel far-field grid, 64 lasers, 24 enemies, 40 items,
and player resources, wall distances, bomb availability and elapsed event times.
Rewards concern damage, kills and stage completion with remaining-life bonuses;
hits and actual bomb activations are penalized. Survival and score are not reward
terms. Updates are described at game-over boundaries. Bomb-period damage needs
special handling to prevent a bombing reward exploit.

Exact coefficients, neural architecture, PPO hyperparameters and complete
observation encoding are not published there. All values below are **local,
provisional design choices**, not recovered Sharu settings.

## Implemented initial components

`configs/sharu-inspired-v1.json` records source policy and provisional settings.
It is deliberately separate from `mock-smoke.json`, which still tests only
infrastructure with zero reward.

`game_policy.py` encodes a Gym Dict of float32 numerical arrays:

| Array | Shape | Local encoding |
| --- | --- | --- |
| player | 15 | Position, velocity, lives, power, four wall distances, bomb availability, elapsed hit/bomb times, movement width/height |
| bullets | 128 x 11 | Mask, relative XY, velocity XY, acceleration XY, radii XY, approach distance/time |
| far_grid | 3 x 112 x 96 | Density and average velocity XY of bullets not in the nearest set |
| lasers | 64 x 7 | Mask, relative XY, angle sine/cosine, length, width |
| enemies | 24 x 8 | Mask, relative XY, dimensions, velocity XY, phase HP fraction |
| items | 40 x 5 | Mask, relative XY, velocity XY |

Coordinates are nominal 384x448 playfield pixels with downward-positive Y;
velocities are pixels/second, acceleration pixels/second squared, laser angles
radians. This nominal canvas is **not** a measured movement boundary. The adapter
must supply verified player bounds; the encoder refuses out-of-bound player
positions. Live calibration remains required, especially at the top wall.

Entities are distance-sorted with stable lifetime IDs for ties; empty slots have
mask zero. Accelerations must be tracked by identity, never by array index.
Approach prediction uses constant relative velocity over a provisional two-second
horizon; it is not exact prediction of curved trajectories or collision geometry.
Far density uses log(1+count)/log(17); averages use all far bullets in each cell.
Features are clipped to [-1,1]. Off-canvas far bullets are omitted from the grid.
Unknown required state must stop collection rather than silently become zero.
At episode start, elapsed timers may use the documented saturation value of
10 seconds until their first event; they are not exact invulnerability timers.

Actions are locally defined as MultiDiscrete([9,2,2,2]): neutral/eight directions,
shoot, focus, bomb. The decoder returns key names but **does not send input**.
No forced shooting, wall punishment, scripted dodging or bomb thresholds are added.

The local network uses masked shared entity MLPs with mean/max pooling, a small
convolution branch for the numerical far-field grid, and a player MLP. Combined
features have size 128; policy and value heads each use two 128-unit layers.
The grid is not a screenshot. CPU, seed 7, learning rate 0.0003, gamma 0.99,
GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, rollout 2048, batch 64,
and 10 epochs are provisional. Standard SB3 fixed-rollout updating is **not**
the source's episode-end collector; that live scheduling is not implemented.

## Reward contract

`game_rewards.py` consumes verified adapter events scoped to one episode.

| Confirmed event | Provisional reward |
| --- | --- |
| Damage | 1 x damage / phase-initial enemy HP |
| Enemy defeat | +0.5 |
| Stage clear | +20 plus 5 x remaining lives |
| Hit/death | -5, independent of event time |
| Actual bomb activation | -0.5 |

Damage while a bomb is active is excluded conservatively; this is our explicit
interpretation, not a claim about the original exact suppression formula. There
is no cumulative damage-reward cap. Repeated event IDs are idempotent, conflicting
duplicates rejected, and kill/stage subjects counted once per episode. Invalid
batches do not consume valid events. The adapter must emit unique authoritative
damage intervals and actual activation/hit events, not infer them from power
loss, a pressed button, disappearing entities or boss timeouts. Enemy lifetime
and boss-phase identities and HP transitions still require verification.

No explicit reward for survival, score, power pickup, wall avoidance or narrator
output is added. Tests cover bomb-period suppression, fixed hit cost, duplicated
events, atomic rejection, observation shapes, ordering, approach calculation and
save/reload of the policy.

## Run the offline diagnostic

From the project directory in PowerShell:

```powershell
./scripts/learning.ps1 policy-check
```

This constructs an **untrained** PPO model, exercises synthetic entity data,
checks finite feature gradients, saves/reloads the model, and records model-only
prediction timings. It never launches the game, contacts the bridge, sends keys,
collects demonstrations or calls PPO.learn. Outputs are in a new
`artifacts/policy-...` directory: `profile.json`, `result.json`, and
`untrained-policy.zip`. Only trusted locally generated models should be loaded.

## Live training gates

Update: [the native diagnostic adapter](live-adapter.md) now supplies raw-state
reads, game-local input, exact stage-1 frame stepping and cold startup/restart.
The profile is still not connected to it: entity identity, HP/reward events,
geometry normalization and full live Gym/HTTP collection remain separate gates.

`live_training_enabled` must remain false. The contract-only environment refuses
reset/step; the existing runner accepts only the mock profile. Neither a synthetic
test nor the earlier read-only memory probe demonstrates gameplay readiness.

Before any live learning, implement and verify:

1. Game-version-specific entity extraction, movement limits, player state and
   damage/kill/stage/hit/bomb events, with stable lifetime and episode IDs.
2. Exactly two game frames per decision, stale-snapshot rejection and a measured
   end-to-end latency budget. No unlocked 30 Hz polling presented as frame sync.
3. Input release on error/stop/focus loss, terminal detection and reliable reset
   into Normal/Reimu B without automating UAC or modifying security settings.
4. Game-over-boundary rollout collection/update, correct bootstrap handling,
   bounded run budget, checkpoints and trustworthy resume behavior.
5. Short human-observed live acceptance tests before unattended training.

Future evaluation should track stage/clear rate, hits, confirmed bombs, damage,
remaining lives, return components and frame/latency faults. No learning success
or 24-hour unattended run is claimed or started by this initial setup.
