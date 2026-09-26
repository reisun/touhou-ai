# Native combat rewards

> Historical v3/v4 record. Later reward revisions are indexed in the
> [model/learning history](model-learning-history.md); the adopted revision is v9 as of 2026-09-26.

Current reward version is `th10-rewards-v4`: damage +3.0 per 1000 HP and
life lost -15.0. Kill, stage-clear and power weights are unchanged. OBS2 labels
read the same backend weights used by the collector. Prior checkpoints remain
preserved and are not silently resumed under the changed reward contract.

Live reward version `th10-rewards-v3` enables damage and kill events in addition
to stage clear, hit and power. Formula weights are unchanged from v2. Old models
and recordings are preserved, but resuming older reward/observation contracts is
rejected. Individual bullet capacity is now 40, with overflow in the full-scene
density/mean-velocity grid.

The pinned Steam executable is checked by SHA-256 and instruction signatures.
The frame-gate observer records the actual HP subtraction at `0x40e1b6`, after
invulnerability and damage-reduction checks. Effective damage is capped by
remaining HP and the active scripted HP threshold; overkill is not rewarded.
The death-score branch at `0x40e231` produces a kill only when the same enemy
received lethal applied damage in that tick. Phase callbacks run before this
branch. Off-screen removal and scripted HP writes do not generate events.

Enemy identities are scoped to the attachment and constructor lifetime, reset on
construction (`0x40da27`) and removed on destruction (`0x40dae0`). Unique event
IDs and reward-layer deduplication prevent repeat awards. Each explicit frame
step resets the event batch; repeated snapshots do not consume it. Hooks only
collect during guarded gameplay, not menus or optimizer holds. Errors and queue
overflow stop observation rather than silently lose events.

## Evidence

`artifacts/combat-acceptance-20260926-a`: 753 two-frame samples, 98 damage events,
27 kills, 1103 effective HP damage. Neutral first 600 frames: zero combat events.
54 independently compared surviving-enemy HP differences matched event sums.
No model updates were performed.

Second run reproduced those counts. Its attempted bomb did not activate (power
was zero), so it is NOT evidence for bomb damage. Boss phase changes, timeout
and final boss death still need real-game acceptance; unit checks cover clipping
at a phase HP floor and kill deduplication, not those real scenes.

Public source consulted as a navigation aid, with addresses and semantics checked
against the running executable (the upstream original binary differs from Steam):
https://github.com/N0zoM1z0/th10/blob/main/src/Enemy.cpp
