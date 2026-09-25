# Line laser collision acceptance

Evidence: `artifacts/laser-boundary-20260926-01/{calls,states}.jsonl` and `status.json`.
Pinned TH10 executable, managed PID 37992, stage 1 Normal, spell ID 7.

The user authorized using the current play and consuming lives. The probe used
ordinary directional/focus input only, with no shooting, bomb, stat edits, model
updates, or synthetic calls to game routines. It stopped on the first life loss
and returned to pause phase 2. Reserve lives changed from 2 to 1.

The trace intercepted the game's own line-laser calls to 0x4267f0 (return address
0x41d6c3) and recorded arguments, player hitbox/status, and the returned result.
The return value 1 invokes the game's hit path; it is not inferred merely from
proximity or a later life decrement. Probe frame range: 7854 through 7904.

## Results

- 1,600 line collision calls; one actual laser hit; zero predicted-hit mismatches.
- Maximum rectangle parameter error: 0.000009523 game pixels/radians.
- Sampled state: 2; raw width: 14; raw lengths: 144 through 180.
- Closest eligible miss margin: +1.74456 pixels; hit margin: -0.237085 pixels.
  These are sampled margins, not a subpixel bisection of the exact boundary.
- Actual hit at frame 7895: player (47.14, 369.01), halfbox (1, 1),
  rectangle origin (32.09805, 227.20583), angle 1.4352173,
  collision length 144 and width 7. The game returned 1.
- Game remained alive and paused after instrumentation was removed.

`collision.field_validated` now applies only to line/state2/raw-width14/raw-length
144..180, the observed scope. This validates the sampled hit/miss classification
and rectangle arguments, not all possible angles/widths/states at arbitrary
precision. Activation/deactivation transitions and other laser types remain
unverified. `activation_transition_validated` remains false.
