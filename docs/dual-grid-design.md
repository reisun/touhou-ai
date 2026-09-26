# Dual-grid observation design

> The opt-in implementation described below was subsequently adopted in live learning.
> See [switch evidence](dual-grid-switch-20260926.md) and [history](model-learning-history.md).

Implemented 2026-09-26 as `th10-dual-grid-v1`, an explicit opt-in architecture.
Existing focused training remains the default; no current campaign was stopped,
restarted, or migrated during implementation. Reward weights are unchanged.

## Observation contract

| Input | Shape | Purpose |
| --- | --- | --- |
| Local grid | 6 x 96 x 96 | Player-centered 192 x 192 game pixels, 2 px cells |
| Global grid | 14 x 56 x 48 | Entire 384 x 448 playfield, fixed 8 px cells |
| Player state | 21 | Existing resources, motion, immunity, bomb/spell state |
| Previous rewards | 7 | Existing previous-completed-action reward input |

Total: 92,956 values. No history, nearest-40 or nearest-10 arrays, individual
enemy/item/laser arrays. All available entities are considered, with no top-N
limit. Collision-enabled ordinary bullets appear in both grids; local selection
uses hitbox/viewport intersection, not just the center being inside the viewport.

Local channels: player coverage, bullet coverage, coverage-weighted mean bullet
velocity X/Y, active laser coverage, field-validated laser coverage.

Global channels: bullet density and mean velocity X/Y; enemy density and mean
velocity X/Y; mean known enemy HP ratio, fraction with known HP, mean known max
HP / 100000; boss density; item density; player location; active laser coverage;
field-validated laser coverage. Enemy/item locations are point bins, not claims
of calibrated body collision shapes. Item types were not part of the previous
five-value item input and are not added here.

Density is log(1+count)/log(17), clipped to 1. Velocity is game pixels per frame
divided by 10, clipped to [-1,1] after averaging. Opposed velocities in one cell
can cancel; this remains a limitation of a single instant with mean velocities.
Unavailable entity managers fail explicitly; they are not encoded as safe empty
space. Unknown enemy HP is represented by its known-fraction channel.

## Geometry evidence and limits

Read-only disassembly of the running, pinned TH10 executable (SHA256
`3BDB72CF3D7C33C183359D368C801490DBCF54E6B3B2F060B95D72250B6866A3`):

- Bullet update at `0x406483` tests bit 2 of the bullet flags. `0x406494` passes
  bullet+0x3f0 to `0x4266b0`, with its position in ECX and player in EDX.
- `0x4266b3..0x4266df` multiplies both bullet extents by 0.5 and builds an
  axis-aligned rectangle about the center. Constant `0x470b0c` was read as 0.5.
  Thus bullet `hitbox_raw` is full width/height, not radius.
- `0x425f00..0x425f86` constructs player min/max coordinates by subtracting and
  adding player+0x41c/0x420 directly. Thus player `hitbox_raw` is half width/height.
- `0x4266df..0x42671c` compares those rectangles. Immunity/status checks follow;
  the grid represents geometry, while immunity/status stay in numeric inputs.
- The reader now exports `flags_raw` for bullets. Old recordings without this
  field cannot silently be used as collision-ready grid observations.

This is binary-derived evidence, not a new live bullet contact trace. The
read-only snapshot checked during this task contained a player with halfbox
[1,1], but no bullets/enemies/items/lasers; it is not live combat validation.
No hooks, game inputs, lives, or model parameters were changed by that check.

Each ordinary rectangle contributes its exact area fraction inside each cell.
Overlapping objects use maximum coverage, **not exact union area**. Laser
rectangles use four subcell samples, so thin or diagonal boundaries are
approximate. Existing laser field-validation limits remain visible through a
separate channel; they are not promoted to universally verified geometry.
Grid discretization and the CNN do not guarantee exact collision avoidance.

## Network and storage

The local CNN starts at full resolution with stride 1, then uses three learned
stride-2 convolutions, preserving a 12 x 12 layout before mapping to 128 features.
The global CNN preserves a 14 x 12 layout and maps to 128 features. Player and
reward branches each produce 32, then 320 -> 256 merges them. Policy/value heads
remain 256 -> 128 followed by the existing 15 action scores / one value.

Grid values undergo a float16 round trip before action selection. Rollout grids
are stored as float16, so PPO receives the same observation values used to sample
the action, with no extra rollout-only quantization. Rewards, GAE, values, action
log probabilities, and model computation remain float32.

Observation storage is 185,968 bytes per transition: about 319 MiB for 1,800
steps and 3.12 GiB for 18,000 steps, excluding model/batches/other buffers. No
claim is made that the new network is cheaper than the old entity-only model.

## Validation and preview

`scripts/check_dual_grid.py --output <new-directory>` never opens the game. It
produces a synthetic scene preview and timing report. The initial result is at
`artifacts/dual-grid-check-20260926/`:

- 934,728 model parameters.
- Full deterministic `model.predict`: median 2.53 ms, p95 2.85 ms.
- 2,000 synthetic bullets encoded: median 6.97 ms, p95 7.84 ms.
- These omit frame-gate, memory reading, logging and real combat; they establish
  neither end-to-end 30 Hz performance nor improved play.

Tests cover fractional boundaries, out-of-view centers with intersecting boxes,
translation invariance, full-scene counts beyond old caps, inactive flags,
unknown HP, missing data rejection, laser layers, scalar/vectorized equivalence,
PPO update and gradient flow, half-storage observation/log-probability consistency,
checkpoint save/reload, and incompatible resume rejection. OBS keeps new and old
architecture learning curves separate and labels its coarse display as a scene
preview rather than the full neural input.

## Selecting the architecture

After coordinating with the current game owner, a new bounded campaign uses:

```powershell
./scripts/live-learning.ps1 rehearse -DualGrid -Episodes 1 -MaxSteps 1800 -MaxSeconds 300 -NoUI
```

This command was **not run** as part of implementation. It controls/restarts the
managed game when actually invoked. Do not use ResumeLatest for an old focused
model. Both the PowerShell preflight and Python manifest validation reject it.
Continuous learning and any current-run switch remain separate operational steps;
begin with a bounded real-game check of the new contract and geometry.
