# Full-scene policy observation

> Historical individual-bullet/overflow-grid contract, superseded by dual-grid.
> See [model/learning history](model-learning-history.md) for the sequence of observation changes.

There is no distance cutoff for bullets or items. Decisions remain every two game
frames (30 Hz). The nearest 40 bullets and 40 items are individual inputs.
Overflow on-screen bullets enter the 4x4-pixel density/mean-velocity grid.

OBS1 shows individual bullets as cyan dots and grid cells as amber squares with
mean-velocity strokes. Overflow bullets are not drawn as individual dots. Items
shown are the same nearest 40 passed to the policy. The original monitor retains
all entities. Grid telemetry is tested against both encoders. Spatial-cutoff
checkpoints cannot resume this scope. No learning was run for this change.
