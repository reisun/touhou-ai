# Next observation design (proposal)

> Historical record of the focused architecture. The subsequent dual-grid switch
> has now been adopted; see [history](model-learning-history.md) and [switch evidence](dual-grid-switch-20260926.md).
> Statements below about the existing/default campaign describe the implementation milestone at that time.

Update 2026-09-26: the subsequent local/global grid design is implemented as an
opt-in contract, documented in [dual-grid-design.md](dual-grid-design.md).
The focused architecture below remains the existing default campaign; no switch
was performed during that implementation.

Recorded: 2026-09-26. Implementation and fresh continuous learning subsequently
authorized by the user. Reward formula unchanged. Old campaign preserved at
artifacts/live-learning-20260926-101422-e4a08b (122 updates).

## Implemented architecture: th10-focused-bullets-v1

| Branch | Input | Processing | Output |
| --- | --- | --- | --- |
| Surrounding bullets | 40 x 8 | per bullet 7 -> 64 -> 64, masked mean/max | 128 |
| Nearest bullets | 10 x 8, duplicated from closest 40 | flatten 80 -> 128 -> 128, no pooling | 128 |
| Enemies | 24 x 9 | per entity 8 -> 32 -> 32, masked mean/max | 64 |
| Items | 40 x 5 | per entity 4 -> 32 -> 32, masked mean/max | 64 |
| Lasers | 64 x 12 | per entity 11 -> 32 -> 32, masked mean/max | 64 |
| Player | 21 | 21 -> 32 | 32 |
| Previous rewards | 7 | fixed bounded scaling, 7 -> 32 | 32 |
| Merge | 512 | 512 -> 256 | 256 |
| Policy | 256 | 256 -> 256 -> 128 | 15 logits |
| Value | 256 | 256 -> 256 -> 128 | 1 |

Raw inputs total 1,612. No grid allocation or CNN in this contract.
Nearest slots are distance-sorted with deterministic tie breaks, not stable IDs.
No history or extrapolated positions are included. OBS1 marks the nearest ten
in amber and the remaining selected bullets in cyan.

Reward order: damage, kill, stage_clear, hit, power, low_power, invalid_bomb.
Fixed scales: 1, 1, 10, 15, 6, 0.021, 7/30. Transform each v to
(v/scale)/(1+abs(v/scale)). Inputs are the previous completed action's actual
components, zero at episode start; reward objective itself is unchanged.
Changing reward weights affects both targets and these inputs. Resume rejects
different reward weights, scope, or architecture. Do not silently migrate.
Passive shadow inference is unavailable without exact previous-action reward
history; live collector inference and OBS remain connected normally.

## Agreed direction for the next implementation

- Remove the far_grid observation and its CNN branch.
- Keep the nearest 40 individual bullets and their current eight input fields.
- Expand only the bullet encoder from 7 -> 32 -> 32 to 7 -> 64 -> 64.
  The presence mask remains separate. Mean and max pooling then produce 128
  features instead of 64. Removing the grid's 64 features leaves the combined
  feature count at 352 before the new reward branch is added.
- Add the previous action's realized reward components as seven inputs:
  damage, kill, stage_clear, hit, power, low_power, invalid_bomb.
- Use zero for absent events and at episode initialization. Feed a completed
  action's components only into the next decision, never the action that caused
  them. Provide the same inputs during training and inference.
- Scale reward inputs using documented fixed scales without changing the reward
  objective. Branch width and exact scales remain to be selected.
- Preserve existing models. The proposed architecture should start a separate
  fresh campaign when implementation and switching are authorized.

## Open question: predicted bullet positions

The user is considering explicit positions predicted at the next decision
(currently two game frames), in addition to current position, velocity, and
estimated acceleration. This is not yet an agreed input addition.

These derived features add an inductive bias, not an independent observation.
Check velocity/acceleration units and game update order before implementing a
predictor. Curving, spawning, and scripted velocity changes limit extrapolation.
Evaluate separately from the baseline architecture change. Relative closest
approach distance/time are alternative candidate features, not approved changes.
