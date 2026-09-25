# TH10 live adapter milestone

Implemented and tested on 2026-09-25. This document describes the original
diagnostic milestone. Subsequent actual PPO learning is documented in
[real-learning.md](real-learning.md); the complete Sharu profile is still pending.

## Research

- [Priw8/sht-webedit](https://github.com/Priw8/sht-webedit), inspected revision
  `98b8cca076c3a42e0f7ab5f9835e801993af727c`, describes static SHT shot data.
  Its TH10 schema includes movement speeds, hitboxes and shot damage. It is useful
  for later calibration, but is not a live enemy/bullet extraction interface.
  No SHT/game data files were modified or uploaded to the web editor.
- [projMiss/Th10Ai](https://github.com/projMiss/Th10Ai), especially Th10Apis.cpp
  and Th10Raws.h, supplies runtime address/structure facts for players, pooled
  bullets/items and linked enemies/lasers. It warns that bullet array slots move
  between frames. Slots are therefore never advertised as stable entity IDs here.
- [binvec/TH10_DataReversing](https://github.com/binvec/TH10_DataReversing), revision
  `e7b3c42d21de49cd22cab3d3da0986d2c4b782ae`, and
  [Infinideastudio/TH10AI](https://github.com/Infinideastudio/TH10AI), revision
  `9388c79051604a3fcb3dc29a2c558cf186aa0d56`, corroborate several runtime fields.
- [OpenInputLagPatch's TH10 implementation](https://github.com/khang06/OpenInputLagPatch/blob/e4eb70ba0b95a4af9994dbc746cebce8d3b7eb0f/openinputlagpatch/games/touhou10.cpp)
  identifies the window update function. Local disassembly of the pinned process
  then identified its update-chain call and keyboard-state update routine.
  The frame limiter patches from that project were **not** installed.
- [thprac's TH10 implementation](https://github.com/touhouworldcup/thprac/blob/master/thprac/src/thprac/thprac_th10.cpp)
  corroborates character, shot, difficulty, stage and input-state addresses.
  No thprac practice/stat modifications were applied.
- [Frida's official API](https://frida.re/docs/javascript-api/) documents the
  temporary interception and native Windows event calls used for synchronization.

These are independently written Python/JavaScript components based on address
facts, local instruction inspection and observed behavior. No external bot,
pretrained weights, practice patches or executable code were copied into the game.
Upstream licenses must be reviewed before importing their implementations.

## What is implemented

`th10_reader.py` reads bounded arrays/lists using the existing ReadProcessMemory
wrapper. Pointers, read sizes, list cycles and finite numerical fields are checked.
An unavailable manager yields null, not an invented empty entity set. Enemy HP is
explicitly null. Bullet slots and raw object addresses are diagnostic identifiers,
not lifetime IDs. Raw coordinates/velocity/hitbox fields are retained without
claiming that all normalization and collision semantics have been calibrated.

In the verified build, native player X is centered: starting X=0 corresponds to
the initial profile's nominal X=192. Native player velocity 450 corresponds to
4.5 pixels/frame, not 450 pixels/second. These conversions must be applied by a
future validated observation adapter; do not feed raw snapshots into game_policy.
`lives_raw` counts reserve lives: 2 initially; -1 was observed at terminal death.

`th10_gate.js` installs temporary Frida hooks in the running game only:

- `0x449c00` is the update-chain entry, reached after the normal limiter. A
  sequence number counts these updates, including menus. The main game thread
  waits on a Windows event at requested boundaries while Python reads a snapshot.
- `0x44a5f0` updates keyboard state. Only primary game input is overridden;
  current/previous/repeat/pressed/released words remain coherent. No SendInput,
  global keyboard injection or desktop focus manipulation is used.
- `LiveRuntime.step_gameplay` additionally checks the **game's own stage frame**
  advanced by the requested amount. Pauses/loading stalls fail instead of being
  mislabeled as two gameplay frames. Stage changes are reported separately.
- A three-second command/park timeout neutralizes input and releases the wait.
  F8 is an emergency release, checked at update boundaries; a parked wait may
  take up to three seconds to release. The fault is latched; a new session is needed.
- Cleanup signals the wait, removes hooks and closes the event. A named Windows
  object refuses a competing controller for the same PID. Restart also checks
  for an existing live owner before closing the managed game.

The Python client validates the exact executable SHA256 and PE32 base before
attachment. The hook checks instruction signatures. Only the configured
SHA256 `3BDB72CF3D7C33C183359D368C801490DBCF54E6B3B2F060B95D72250B6866A3`
is currently supported. Another build/mod must be investigated separately;
changing a config hash alone cannot enable it.

## Automatic startup and restart

`live_reset.py` observes the menu object at 0x47784c. The locally verified fields
are screen at +0x1c, phase at +0x20, selection at +0x24 and count at +0x2c.
It waits through transitions and chooses Normal/Reimu B through ordinary game
input, without writing difficulty, lives, power, stage or player position.
Unknown menu layouts fail. Startup waits for an actionable player and verifies
stage 1, replay mode 0, normal mode flags and initial reserve lives.

`scripts/live.ps1` reuses the existing PID/start-time/path-checked normal stop/start
scripts. Restart is a **cold process restart**, not an in-memory savestate or a
fast pause-menu retry. This is slower but exercises normal game initialization.
The bounded episode loop restarts after verified terminal reserve-life exhaustion
or the explicitly configured diagnostic step cap. It does not retry arbitrary
errors, continue after frame faults, overwrite existing run files or run forever.

## Commands

Install the pinned host-only dependency in this project's virtual environment:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-windows-live.txt
```

With Steam signed in and window-mode prompts disabled:

```powershell
# Current managed game: neutral-input diagnostic, no restart.
./scripts/live.ps1 probe -Steps 30

# Restart, enter Normal/Reimu B, collect a bounded run; leave game running.
./scripts/live.ps1 restart -Steps 150

# Two bounded acceptance episodes, including restart and cleanup.
./scripts/live.ps1 verify -Steps 900 -Episodes 2
```

`verify` checks horizontal displacement, focus state and actual input words,
then collects neutral-action frames (not a learned policy). It deliberately
withholds a command to test the watchdog and stops the managed game on exit.
Avoid manually operating the game during verification. Steps are limited to
1..1800 and episodes to 1..3. `probe` cannot restart episodes. Diagnostics contain
raw game data in `artifacts/live/*.jsonl`, plus `.status.json` results.

`input_mask([direction, shoot, focus, bomb])` maps the initial MultiDiscrete
contract to the native bits. Bomb input is implemented but actual bomb activation
and its event accounting have not been acceptance-tested with sufficient power.

## Evidence

Two consecutive normal-stage runs, with neutral actions after the input checks:

- `artifacts/live/20260925-223315-bfc584.status.json`
- `artifacts/live/20260925-223357-6e491a.status.json`

Each reached terminal reserve lives at stage frame 1506 and completed 733 recorded
main-loop observations (initial observation plus 732 two-frame steps), in addition
to input-test observations. Both ended with automatic input release and a passed
watchdog check. The next run started automatically in a newly verified process.
Observed maxima were 40 bullets, 11 enemies and 25 items. No laser appeared;
laser parsing has synthetic coverage but is **not live-validated**.

`artifacts/live/20260925-223646-1b98a0.status.json` adds explicit focus and native
input-word checks, including zero native input following the watchdog.
Horizontal movement was +45 then -45 native pixels over ten frames each.
These are acceptance fixtures, not scores or evidence of learning skill.

## Remaining training gates

The following must remain disabled/unverified before connecting the PPO policy:

1. Stable entity lifetime IDs, bullet acceleration and verified collision masks.
2. Enemy/boss HP and phase identities; authoritative damage, kill, stage-clear,
   hit and bomb events. Do not infer kill rewards from disappearance or bomb
   activations from a command or power decrease.
3. Live laser geometry and calibrated player movement limits/normalization.
4. Stage-transition handling beyond stage 1 and the full episode-end PPO
   collection/update loop with bounded training budgets.
5. A live HTTP/Gym adapter with explicit capabilities. The existing HTTP bridge
   remains the mock backend; these diagnostics are a native Python interface.

`live_training_enabled` stays false and the existing mock learner is unchanged.
The host dependency was added only to .venv; no Windows security/UAC setting,
driver, game executable, or frame-speed patch was changed.
