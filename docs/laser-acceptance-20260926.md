# Paused live laser acceptance

Captured from the managed TH10 process (PID 37992), without injection or input.
Raw evidence: `artifacts/laser-paused-acceptance-20260926/samples.jsonl`.

- Stage 1, Normal, stage frame 7854, pause phase 2, replay mode 0.
- Boss HP 1667 / 8200; spell ID 7, flags 3.
- 32 line lasers, state 2, length approximately 173.5999, visible-width field 14.
- Binary-derived collision rectangles: length approximately 138.8799, width 7,
  origin offset by 10 percent of the raw length along the laser angle.
- All 32 entities encoded successfully into `th10-live-observed-v2`.
- Existing checkpoint `live-learning-20260926-004328-0a1766/real-episode-2.zip`
  loaded and migrated in memory; inference on this actual snapshot succeeded.
  No model saved, no PPO update, no predicted action sent to the game.
- Windows regression suite: 47 tests, passed with one platform-specific skip.

This verifies live field acquisition and the observation-to-model interface.
It does NOT verify collision boundaries through contact, activation transitions,
or infinite lasers. `collision.field_validated` remains false deliberately.
The pause screen obscures the playfield, so a screenshot alone cannot establish
alignment between rendered beams and computed rectangles.

The game was left paused. Further collision tests must be bounded and distinguish
visual geometry, binary-derived hit geometry, and actual hit outcomes.
