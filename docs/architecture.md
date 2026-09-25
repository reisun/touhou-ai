# Architecture and handoff

Windows owns operations, the game process, the mock bridge, and read-only probe.
Docker Desktop / WSL owns isolated testing, diagnostic learning and evaluation.
Source of truth: D:\repos\touhou-ai. No second editable WSL copy is needed yet.

## Protocol v1

All requests require `Authorization: Bearer <BRIDGE_TOKEN>`.

| Route | Input | Output |
| --- | --- | --- |
| GET /health | None | status, backend, protocol |
| POST /reset | {} | frame, x, y, terminated, backend |
| POST /step | action, frames, expected_frame | same observation |
| POST /shutdown | None | status; terminates bridge |

Actions: 0 neutral, 1 left, 2 right, 3 up, 4 down. Frames: 1..8.
Coordinates are normalized to 0..1. Each mock episode lasts 600 frames.
Stale expected_frame values are rejected without advancing the game.
Mutations are serialized; this initial API supports a single episode owner.
The diagnostic Gymnasium adapter returns zero reward; mock motion is not a
gameplay training target. The runner rejects other reward/backend selections.
Real-time input cannot be claimed frame-synchronous until measured in the game.

## Next milestone

1. Completed: locate/version-check the game, launch/stop, and verify VM_READ.
2. Implement a Windows adapter for observations, input release on failure,
   frame synchronization, terminal detection, and restart.
3. Measure correspondence between actions and observed frames on the real game.
4. Agree real observation/reward definitions and connect to the existing runner.
5. Run a short approved learning/evaluation experiment before extended runs.

Do not distribute game files, download unofficial copies, assume memory offsets,
or use the mock smoke result as evidence that real-game control works.
