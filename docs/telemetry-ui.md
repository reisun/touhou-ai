# Learning monitor and telemetry v1

Start: `./scripts/dashboard.ps1 start`. Status: `./scripts/dashboard.ps1 status`.
Stop only the managed observer: `./scripts/dashboard.ps1 stop`.
Default URL: http://127.0.0.1:18767 (the launcher selects a free port if occupied).

## Reference and scope

The supplied screenshots inform the numerical field, measured inputs, probability
heads, value estimate, reward breakdown and episode history layout. They do not
establish the original neural architecture or exact reward coefficients. Existing
`configs/sharu-inspired-v1.json` remains the provisional local learning policy;
the screenshot's power-change rewards were not silently adopted.

The monitor is a read-only observer, not a game controller or training launcher.
The live panel now also renders actual policy probabilities, value estimates,
life-loss rewards and learner counters from the bounded real-game collector.
The history tab lists these runs separately from mock runs. See `real-learning.md`.
Recording replay initially uses genuine diagnostic game observations, not trained
agent gameplay. No character artwork or game screenshot assets are copied.

## Interface

`touhou_ai.telemetry.packet` converts a raw reader snapshot to `schema_version: 1`.
`episode_id` scopes a recording; `sequence` is the gate tick; `game_frame` is the
stage frame. `source` distinguishes `recording` and `live`. `timestamp` is the live
publication time in Unix seconds; legacy recordings have null timestamps.

Native centered X is shifted by 192 to a nominal 384 x 448 coordinate field.
Native power is divided by 20. Stage seconds are derived as frame / 60.
`lives_reserve` preserves the raw reserve count, including terminal -1.
Velocity and geometry retain `_raw` suffixes: their units and semantics must not
be assumed equivalent to the learner encoder. The nearest bullet-center distance
is geometric, not a collision radius or risk probability. The player ring is a
display marker, not its hitbox. Entity glyph sizes are illustrative.

Missing lists are null, while measured empty lists are empty arrays. Missing input
words are not inferred from player motion. `policy`, `reward` and `learning` stay
null on real observations until those producers are integrated and validated.
Capabilities explicitly flag unresolved IDs, enemy HP, reward events and lasers.

`live_runtime` atomically publishes `artifacts/telemetry/latest.json` while recording
main steps. The UI polls at 500 ms and marks observations older than 2.5 seconds
as stopped / last observation. This is monitoring, not a 30 Hz browser control loop.

Read-only endpoints: `/api/health`, `/api/catalog`, `/api/profile`,
`/api/frame?id=...&index=...`, `/api/series?id=...`, `/api/live`,
`/api/policy-diagnostic`. Recording paths are confined to `artifacts/live`.
The server binds loopback, checks Host, exposes no credentials or control routes,
and does not enable CORS. Do not publish it through a network proxy.

## Model interface

The existing numerical encoder feeds entity pooling, a far-field grid CNN and
player features to PPO actor/value heads. Actions use MultiDiscrete([9,2,2,2]):
direction, shoot, focus, bomb. Direction order is explicit in exported telemetry.

`scripts/learning.ps1 policy-check` exports `policy-telemetry.json` with actual
categorical probabilities and V(s) from the saved/reloaded, untrained model on a
synthetic observation. These appear only in the separate model diagnostic view.
They are neither confidence scores nor evidence of gameplay ability. Distribution
API: https://stable-baselines3.readthedocs.io/en/v2.7.0/common/distributions.html

Live training stays disabled until stable entity lifetimes, velocity conversion,
enemy HP/phase identity, confirmed reward events and laser geometry are validated
and connected to the encoder and reward calculator. Reward bars remain unknown,
not zero. History charts show observed stage frames, not invented episode returns.

## Verification

Unit tests cover coordinate conversion, unknown values, finite positions, confined
read-only routes and actual diagnostic probability normalization. Browser checks
cover desktop/mobile layouts, nonblank canvas, replay controls and disconnected
live state. Local screenshots are in `artifacts/dashboard-*.png`.
